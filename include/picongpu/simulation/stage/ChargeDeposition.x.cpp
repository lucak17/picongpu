/* Copyright 2024 Luca Pennati
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "picongpu/simulation/stage/ChargeDeposition.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/FieldRho.hpp"
#include "picongpu/fields/FieldRhoOperations.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetShape.hpp"
#include "picongpu/particles/particleToGrid/FilteredDerivedAttribute.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.def"


#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>
#include <iostream>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                namespace deriveField = picongpu::particles::particleToGrid;
                template<typename T_SpeciesType, typename T_Area>
                struct ChargeDepositionDetail
                {
                    
                    /** Compute charge density created by a species in an area */
                    HINLINE void operator()(const uint32_t currentStep, FieldTmp& fieldTmp, FieldRho& fieldRho, pmacc::DataConnector& dc)
                    {
                        using SpeciesType = T_SpeciesType;
                        using FilterAll = picongpu::particles::filter::All;
                        using ChargeDensityDerived = deriveField::derivedAttributes::ChargeDensity;
                        using shapeType = typename GetShape<SpeciesType>::type;
                        using Solver = deriveField::ComputeGridValuePerFrame<shapeType, ChargeDensityDerived>;
                        
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail START charge species "<< SpeciesType::FrameType::getName() << " step " << currentStep <<std::endl;
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value device before {24,32,48} "<< fieldTmp.getDeviceDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value host before {24,32,48} "<< fieldTmp.getHostDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                        auto event = deriveField::ComputeFieldValue<CORE+BORDER, Solver, SpeciesType, picongpu::particles::filter::All>()(fieldTmp, currentStep,1u);
                        // wait for unfinished asynchronous communication
                        if(event.has_value())
                            eventSystem::setTransactionEvent(*event);

                        // add charge density fo FieldRho
                        modifyFieldRhoByField<CORE + BORDER, pmacc::math::operation::Add>(fieldRho, fieldRho.getCellDescription(), fieldTmp);
                        /* copy data to host that we can write same to disk probably useless*/
                        fieldTmp.getGridBuffer().deviceToHost();
                        fieldRho.getGridBuffer().deviceToHost();
                        /*## finish update field ##*/
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value device after {24,32,48} "<< fieldTmp.getDeviceDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value host after {24,32,48} "<< fieldTmp.getHostDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                        //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail END "<< currentStep <<std::endl;
                    }
                };

            } // namespace detail

            void ChargeDeposition::operator()(uint32_t const currentStep) const
            {
                using namespace pmacc;
                //using namespace particles::particleToGrid;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& fieldTmp = *dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
                auto& fieldRho = *dc.get<FieldRho>(FieldRho::getName());

                FieldRho::ValueType zeroRho(FieldRho::ValueType::create(0._X));
                fieldRho.assign(zeroRho);

                using SpeciesWithChargeSolver = typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, chargeRatio<>>::type;
                
                meta::ForEach<SpeciesWithChargeSolver,picongpu::simulation::stage::detail::ChargeDepositionDetail<boost::mpl::_1, pmacc::mp_int<type::CORE + type::BORDER>>>depositCharge;

                depositCharge(currentStep, fieldTmp, fieldRho, dc);
                
                //auto* ptrTmp=fieldTmp.getGridBufferPointer();
                //auto* ptrRho=fieldRho.getGridBufferPointer();
                // std::swap(*ptrTmp, *ptrRho);
                //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value device after {24,32,48} "<< fieldTmp.getDeviceDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail tmp value host after {24,32,48} "<< fieldTmp.getHostDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail Rho value device after {24,32,48} "<< fieldRho.getDeviceDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                //std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp/detail Rho value host after {24,32,48} "<< fieldRho.getHostDataBox()({24,32,48}) << " Step " << currentStep <<std::endl;
                std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp END step "<< currentStep <<std::endl;
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu