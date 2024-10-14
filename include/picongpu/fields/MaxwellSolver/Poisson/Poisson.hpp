/* Copyright 2013-2024 Luca Pennati
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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.hpp"

#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/Poisson/Poisson.def"
#include "picongpu/fields/MaxwellSolver/Poisson/Poisson.kernel"
#include "picongpu/fields/cellType/Yee.hpp"
#include "picongpu/traits/GetMargin.hpp"
//#include "include/picongpu/particles/particleToGrid/ComputeFieldValue.hpp"


#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/types.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include <memory>
#include <cstdint>
#include <limits>
#include <iostream>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            class Poisson : public ISimulationData
            {
            public:
                using SuperCellSize = MappingDesc::SuperCellSize;
                using CellType = cellType::Yee;

                Poisson(MappingDesc const cellDescription):cellDescription(cellDescription)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    fieldE = dc.get<FieldE>(FieldE::getName());
                    fieldB = dc.get<FieldB>(FieldB::getName());
                    fieldV = dc.get<FieldV>(FieldV::getName());
                    fieldRho = dc.get<FieldRho>(FieldRho::getName());
                }

                void update_beforeCurrent(uint32_t const currentStep)
                {   
                    float3_X valueB={1.0,2.0,3.5};
                    float_X valueRho=1.5;
                    
                    fieldB->assign(valueB);

                    GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
                    const SubGrid<simDim>& subGrid = pmacc::Environment<simDim>::get().SubGrid();
                    //fieldRho->assign(valueRho);
                    //setFieldBConstantValue<CORE + BORDER>(valueB, currentStep);
                    //setFieldRhoConstantValue<CORE + BORDER>(valueRho, currentStep);
                    //std::cout<< "Debug in include/picongpu/fields/MaxwellSolver/Poisson/Poisson.hpp/update_beforeCurrent step "<< currentStep <<std::endl;
                }

                void SetBCs(GridController<simDim>& gc)
                {

                }

                template<uint32_t T_area>
                void addCurrent()
                {
                }

                void update_afterCurrent(uint32_t)
                {
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "poisson");
                    return propList;
                }

                static std::string getName()
                {
                    return "FieldSolverPoisson";
                }

                /**
                 * Synchronizes simulation data, meaning accessing (host side) data
                 * will return up-to-date values.
                 */
                void synchronize() override{};

                /**
                 * Return the globally unique identifier for this simulation data.
                 *
                 * @return globally unique identifier
                 */
                SimulationDataId getUniqueId() override
                {
                    return getName();
                }
            
            
            
            private:
                template<uint32_t T_Area>
                void setFieldBConstantValue(float3_X const valueB, float_X const currentStep)
                {
                    using Kernel = fdtdPoisson::KernelUpdateField;
                    auto const mapper = pmacc::makeAreaMapper<T_Area>(cellDescription);

                    PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                fdtdPoisson::setFieldBConstantValueFunctor{},
                                fieldB->getDeviceDataBox(),valueB);
                }
                
                template<uint32_t T_Area>
                void setFieldRhoConstantValue(float_X const valueRho, float_X const currentStep)
                {
                    using Kernel = fdtdPoisson::KernelUpdateField;
                    auto const mapper = pmacc::makeAreaMapper<T_Area>(cellDescription);

                    PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                fdtdPoisson::setFieldRhoConstantValueFunctor{},
                                fieldRho->getDeviceDataBox(),valueRho);
                }

                MappingDesc const cellDescription;
                std::shared_ptr<FieldE> fieldE;
                std::shared_ptr<FieldB> fieldB;
                std::shared_ptr<FieldV> fieldV;
                std::shared_ptr<FieldRho> fieldRho;
               // std::unique_ptr<FieldTmp> fieldTmp;
            
            
            
            
            
            }; // class Poisson

            /** Specialization of the CFL condition checker for the Poisson solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<Poisson, T_Defer>
            {
                /** No limitations for this solver, allow any dt
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    return std::numeric_limits<float_X>::infinity();
                }
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for any field access in the None solver
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct GetMargin<picongpu::fields::maxwellSolver::Poisson, T_Field>
        {
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 2>::type;
            using UpperMargin = LowerMargin;
        };
    } // namespace traits

} // namespace picongpu
