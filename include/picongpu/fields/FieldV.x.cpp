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



#include "picongpu/fields/FieldV.hpp"

#include "picongpu/simulation_defines.hpp"
#include "picongpu/simulation_types.hpp"

#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
//#include "picongpu/particles/filter/filter.hpp"
//#include "picongpu/particles/traits/GetInterpolation.hpp"
//#include "picongpu/particles/traits/GetMarginPusher.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

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
 
    /** Create a field
     *
     * @param cellDescription mapping for kernels
     * @param id unique id
     */

    /** Create a field
     *
     * @param cellDescription mapping for kernels
     */


    FieldV::FieldV(MappingDesc const& cellDescription):SimulationFieldHelper<MappingDesc>(cellDescription),id(getName())
    {
        buffer = std::make_unique<Buffer>(cellDescription.getGridLayout());

        // @todo fix margins
        
        DataSpace<simDim> originGuard = GetLowerMargin<fields::Solver, FieldV>::type::toRT();
        DataSpace<simDim> endGuard = GetUpperMargin<fields::Solver, FieldV>::type::toRT();
        //GetMargin<picongpu::fields::, T_Field>
        //math::Vector<int, simDim> originGuard = {1,1,1};
        //math::Vector<int, simDim> endGuard = {1,1,1};

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
    typename FieldV::Buffer& FieldV::getGridBuffer()
    {
        return *buffer;
    }

    //! Get the grid layout
    GridLayout<simDim> FieldV::getGridLayout()
    {
        return cellDescription.getGridLayout();
    }

    //! Get the host data box for the field values
    typename FieldV::DataBoxType FieldV::getHostDataBox()
    {
        return buffer->getHostBuffer().getDataBox();
    }

    //! Get the device data box for the field values
    typename FieldV::DataBoxType FieldV::getDeviceDataBox()
    {
        return buffer->getDeviceBuffer().getDataBox();
    }

    /** Start asynchronous communication of field values
     *
     * @param serialEvent event to depend on
     */
    EventTask FieldV::asyncCommunication(EventTask serialEvent)
    {
        EventTask eB = buffer->asyncCommunication(serialEvent);
        return eB;
    }

    void FieldV::assign(FieldV::ValueType value)
    {
        buffer->getDeviceBuffer().setValue(value);
    }

    /** Reset the host-device buffer for field values
     *
     * @param currentStep index of time iteration
     */
    void FieldV::reset(uint32_t currentStep)
    {
        buffer->getHostBuffer().reset(true);
        buffer->getDeviceBuffer().reset(false);
    }

    //! Synchronize device data with host data
    void FieldV::syncToDevice()
    {
        buffer->hostToDevice();
    }

    //! Synchronize host data with device data
    void FieldV::synchronize()
    {
        buffer->deviceToHost();
    }

    //! Get id
    pmacc::SimulationDataId FieldV::getUniqueId()
    {
        return id;
    }
    
    //Get units of field components
    FieldV::UnitValueType FieldV::getUnit()
    {
        return UnitValueType{sim.unit.eField() * sim.unit.length()};
    }

    std::vector<float_64> FieldV::getUnitDimension()
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

    std::string FieldV::getName()
    {
        return "V";
    }

} // namespace picongpu
