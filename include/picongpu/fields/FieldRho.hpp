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

#include "picongpu/defines.hpp"
#include "picongpu/fields/Fields.def"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    using namespace pmacc;
    /** Representation of the scalar field for total charge density deposited on the grid by particles
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldRho
        : public SimulationFieldHelper<MappingDesc>
        , public ISimulationData
    {
    public:
        //! Type of each field value
        using ValueType = float1_X;

        //! Unit type of field components
        using UnitValueType = promoteType<float_64, ValueType>::type;

        //! Number of components of ValueType, for serialization
        static constexpr int numComponents = ValueType::dim;

        using Buffer = GridBuffer<ValueType, simDim>;

        //! Size of supercell
        using SuperCellSize = MappingDesc::SuperCellSize;

        //! Type of data box for field values on host and device
        using DataBoxType = DataBox<PitchedBox<ValueType, simDim>>;

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */
        
        FieldRho(MappingDesc const& cellDescription);

        //! Destroy a field
        ~FieldRho() override = default;

        //! Get a reference to the host-device buffer for the field values
        GridBuffer<ValueType, simDim>& getGridBuffer();

        GridBuffer<ValueType, simDim>* getGridBufferPointer();

        //! Get the grid layout
        GridLayout<simDim> getGridLayout();

        //! Get the host data box for the field values
        DataBoxType getHostDataBox();

        //! Get the device data box for the field values
        DataBoxType getDeviceDataBox();

        /** Assign the given value to elements
         *
         * @param value value to assign all elements to
         */
        void assign(ValueType value);

        /** Reset the host-device buffer for field values
         *
         * @param currentStep index of time iteration
         */
        void reset(uint32_t currentStep) override;

        //! Synchronize device data with host data
        void syncToDevice() override;

        //! Synchronize host data with device data
        void synchronize() override;


        //! Get unit of field components
        static UnitValueType getUnit();

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */

        static std::vector<float_64> getUnitDimension();


        //! Get mapping for kernels
        MappingDesc getCellDescription();

        //! Get text name
        static std::string getName();

        //! Get id
        pmacc::SimulationDataId getUniqueId() override;
        

        /** Start asynchronous send of field values
         *
         * Add data from the local guard of the GPU to the border of the neighboring GPUs.
         * This method can be called before or after asyncCommunicationGather without
         * explicit handling to avoid race conditions between both methods.
         *
         * @param serialEvent event to depend on
         */
        virtual EventTask asyncCommunication(EventTask serialEvent);

        /** Gather data from neighboring GPUs
         *
         * Copy data from the border of neighboring GPUs into the local guard.
         * This method can be called before or after asyncCommunication without
         * explicit handling to avoid race conditions between both methods.
         */
        EventTask asyncCommunicationGather(EventTask serialEvent);

        /** Bash particles in a direction.
         * Copy all particles from the guard of a direction to the device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void bashField(uint32_t exchangeType);

        /** Insert all particles which are in device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void insertField(uint32_t exchangeType);
        

    private:
        //! Host-device buffer for current density values
        std::unique_ptr<Buffer> buffer;

        //! Buffer for receiving near-boundary values
        std::unique_ptr<Buffer> bufferRecv;

        pmacc::SimulationDataId id;

        //! Events for communication
        EventTask m_scatterEv;
        EventTask m_gatherEv;

        //! Tags for communication
        uint32_t m_commTagScatter;
        uint32_t m_commTagGather;
    };

} // namespace picongpu