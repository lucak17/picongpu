/* Copyright 2013-2023 Axel Huebl, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Pawel Ordyna
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/Fields.def"
#include "picongpu/fields/FieldTmp.hpp"


#include "picongpu/traits/SIBaseUnits.hpp"

/**
 #include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/simulation_types.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
*/

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    /** Representation of the temporary scalar field for plugins and temporary
     *  particle data mapped to grid (charge density, energy density, etc.)
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldRho : public FieldTmp
    {
    public:
        using FieldTmp::FieldTmp;


        FieldRho(MappingDesc const& cellDescription) : FieldTmp(cellDescription, 0), id(getName())
        {
        }


        static UnitValueType getUnit()
        {
            return UnitValueType{sim.unit.charge() / ( sim.unit.length() * sim.unit.length() * sim.unit.length() )};
        }

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        static std::vector<float_64> getUnitDimension()
        {
        /* Rho is in coulombs per cubic meters: C / m^3 = A * s / m^3
         *   -> I * T * L^-3
         */
            std::vector<float_64> unitDimension(7, 0.0);
            unitDimension.at(SIBaseUnits::length) = -3.0;
            unitDimension.at(SIBaseUnits::time) = 1.0;
            unitDimension.at(SIBaseUnits::electricCurrent) = 1.0;
            return unitDimension;
        }

        //! Get text name
        static std::string getName()
        {
            return "Rho";
        }

        /** 
        std::string getId()
        {
            return std::to_string(m_slotId);
        }
        */    
    private:
        //! Host-device buffer for current density values
        std::unique_ptr<GridBuffer<ValueType, simDim>> fieldTmp;

        //! Buffer for receiving near-boundary values
        std::unique_ptr<GridBuffer<ValueType, simDim>> fieldTmpRecv;

        //! Index of the temporary field
        uint32_t m_slotId;

        pmacc::SimulationDataId id;

        //! Events for communication
        EventTask m_scatterEv;
        EventTask m_gatherEv;

        //! Tags for communication
        uint32_t m_commTagScatter;
        uint32_t m_commTagGather;
    };

} // namespace picongpu
