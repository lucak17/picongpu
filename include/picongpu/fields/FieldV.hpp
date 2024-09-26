/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov
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
// from EMFieldBase.hpp
#include "picongpu/defines.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>


// from EMFieldBase.x.cpp
#include "picongpu/simulation_defines.hpp"

//#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
//#include "picongpu/particles/filter/filter.hpp"
//#include "picongpu/particles/traits/GetInterpolation.hpp"
//#include "picongpu/particles/traits/GetMarginPusher.hpp"
//#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

// from EField.hpp
#include <pmacc/algorithms/PromoteType.hpp>

// from EField.x.cpp
#include "picongpu/simulation_types.hpp"
//#include "picongpu/traits/GetMargin.hpp"


#include <type_traits>
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <iostream>






namespace picongpu
{
    /** Representation of the electric field
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldV
        : public SimulationFieldHelper<MappingDesc>
        , public ISimulationData
    {
    public:

        //! Type of each field value
        using ValueType = float1_X;

        //! Number of components of ValueType, for serialization
        static constexpr int numComponents = ValueType::dim;

        //! Type of host-device buffer for field values
        using Buffer = pmacc::GridBuffer<ValueType, simDim>;

        //! Type of data box for field values on host and device
        using DataBoxType = pmacc::DataBox<PitchedBox<ValueType, simDim>>;

        //! Size of supercell
        using SuperCellSize = MappingDesc::SuperCellSize;

        //! Unit type of field components
        using UnitValueType = promoteType<float_64, ValueType>::type;

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         * @param id unique id
         */

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */


        FieldV(MappingDesc const& cellDescription):SimulationFieldHelper<MappingDesc>(cellDescription),id(getName())
        {
            buffer = std::make_unique<Buffer>(cellDescription.getGridLayout());

            // @todo fix margins
            math::Vector<int, simDim> originGuard = {1,1,1};
            math::Vector<int, simDim> endGuard = {1,1,1};

            auto const commTag = pmacc::traits::getUniqueId<uint32_t>();

            for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
            {
                DataSpace<simDim> relativeMask = Mask::getRelativeDirections<simDim>(i);
                /* guarding cells depend on direction
                 * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                 * don't switch end and origin because this is a read buffer and no send buffer
                 */
                DataSpace<simDim> guardingCells;
                for(uint32_t d = 0; d < simDim; ++d)
                    guardingCells[d] = (relativeMask[d] == -1 ? originGuard[d] : endGuard[d]);
                buffer->addExchange(GUARD, i, guardingCells, commTag);
            }

        }

        

        //! Get a reference to the host-device buffer for the field values
        Buffer& getGridBuffer()
        {
            return *buffer;
        }

        //! Get the grid layout
        GridLayout<simDim> getGridLayout()
        {
            return cellDescription.getGridLayout();
        }

        //! Get the host data box for the field values
        DataBoxType getHostDataBox()
        {
            return buffer->getHostBuffer().getDataBox();
        }

        //! Get the device data box for the field values
        DataBoxType getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        /** Start asynchronous communication of field values
         *
         * @param serialEvent event to depend on
         */
        EventTask asyncCommunication(EventTask serialEvent)
        {
            EventTask eB = buffer->asyncCommunication(serialEvent);
            return eB;
        }

        /** Reset the host-device buffer for field values
         *
         * @param currentStep index of time iteration
         */
        void reset(uint32_t currentStep) override
        {
            buffer->getHostBuffer().reset(true);
            buffer->getDeviceBuffer().reset(false);
        }

        //! Synchronize device data with host data
        void syncToDevice() override
        {
            buffer->hostToDevice();
        }

        //! Synchronize host data with device data
        void synchronize() override
        {
            buffer->deviceToHost();
        }

        //! Get id
        pmacc::SimulationDataId getUniqueId() override
        {
            return id;
        }
        
        //Get units of field components
        UnitValueType getUnit()
        {
            return UnitValueType{sim.unit.eField() * sim.unit.length()};
        }

        std::vector<float_64> getUnitDimension()
        {
            /* V is in volts: V  = kg * m^2 / (A * s^3)
            *   -> L^2 * M * T^-3 * I^-1
            */
            std::vector<float_64> unitDimension(7, 0.0);
            unitDimension.at(SIBaseUnits::length) = 2.0;
            unitDimension.at(SIBaseUnits::mass) = 1.0;
            unitDimension.at(SIBaseUnits::time) = -3.0;
            unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;
            return unitDimension;
        }

        std::string getName()
        {
            return "V";
        }

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */        

    private:
        //! Host-device buffer for field values
        std::unique_ptr<Buffer> buffer;

        //! Unique id
        pmacc::SimulationDataId id;
};

} // namespace picongpu
