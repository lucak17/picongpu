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


#include "picongpu/simulation/stage/ChargeDeposition.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/FieldRho.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/filter/filter.hpp"

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
                template<typename T_SpeciesType, typename T_Area>
                struct ChargeDeposition
                {
                    using SpeciesType = T_SpeciesType;
                    using FrameType = typename SpeciesType::FrameType;

                    /** Compute current density created by a species in an area */
                    HINLINE void operator()(const uint32_t currentStep, FieldTmp& fieldTmp, FieldRho& fieldRho, pmacc::DataConnector& dc)
                    {
                        auto species = dc.get<SpeciesType>(FrameType::getName());

                        /* Current deposition logic (for all schemes we implement) requires that a particle cannot pass
                         * more than a cell in a time step. For 2d this concerns only steps in x, y. This check is same
                         * as in particle pusher, but we do not require that pusher and current deposition are both
                         * enabled for a species, so check in both places.
                         */
                    }
                };
            } // namespace detail

            void ChargeDeposition::operator()(uint32_t const step) const
            {
                using namespace pmacc;
                using namespace particles::particleToGrid;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& fieldTmp = *dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
                auto& fieldRho = *dc.get<FieldRho>(FieldRho::getName());

                FieldRho::ValueType zeroRho(FieldRho::ValueType::create(0._X));
                fieldRho.assign(zeroRho);

               // using SpeciesWithChargeSolver = typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, charge<>>::type;
               using SpeciesWithChargeSolver = VectorAllSpecies;

                meta::ForEach<SpeciesWithChargeSolver,detail::ChargeDeposition<boost::mpl::_1, pmacc::mp_int<type::CORE + type::BORDER>>> depositCharge;

                depositCharge(step, fieldTmp, fieldRho, dc);

                std::cout<<  "Debug in include/picongpu/simulation/stage/ChargeDeposition.hpp step "<< step <<std::endl;
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu