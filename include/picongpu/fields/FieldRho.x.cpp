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




#include "picongpu/fields/FieldRho.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldRho.kernel"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/param/fileOutput.param"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <memory>
#include <string>


namespace picongpu
{
    using namespace pmacc;

    template<typename A, typename B>
    using SpeciesLowerMarginOp =
        typename pmacc::math::CT::max<A, typename GetLowerMargin<typename GetInterpolation<B>::type>::type>::type;
    template<typename A, typename B>
    using SpeciesUpperMarginOp =
        typename pmacc::math::CT::max<A, typename GetUpperMargin<typename GetInterpolation<B>::type>::type>::type;

    template<typename A, typename B>
    using FieldRhoLowerMarginOp = typename pmacc::math::CT::max<A, typename GetLowerMargin<B>::type>::type;
    template<typename A, typename B>
    using FieldRhoUpperMarginOp = typename pmacc::math::CT::max<A, typename GetUpperMargin<B>::type>::type;
    /** Create a field
     *
     * @param cellDescription mapping for kernels
     */
    
    FieldRho::FieldRho(MappingDesc const& cellDescription) :  SimulationFieldHelper<MappingDesc>(cellDescription), id(getName())
    {
        m_commTagScatter = pmacc::traits::getUniqueId();
        m_commTagGather = pmacc::traits::getUniqueId();

        buffer = std::make_unique<Buffer>(cellDescription.getGridLayout());

        if(fieldTmpSupportGatherCommunication)
            bufferRecv = std::make_unique<Buffer>(buffer->getDeviceBuffer(), cellDescription.getGridLayout());

        /** \todo The exchange has to be resetted and set again regarding the
         *  temporary "Fill-"Functor we want to use.
         *
         *  Problem: buffers don't allow "bigger" exchange during run time.
         *           so let's stay with the maximum guards.
         */
        const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout().sizeWithoutGuardND();

        using VectorSpeciesWithInterpolation = typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, interpolation<>>::type;
         /* ------------------ lower margin  ----------------------------------*/
        using SpeciesLowerMargin = pmacc::mp_fold<VectorSpeciesWithInterpolation, typename pmacc::math::CT::make_Int<simDim, 0>::type, SpeciesLowerMarginOp>;

        using FieldSolverLowerMargin = GetLowerMargin<fields::Solver>::type;

        using LowerMargin = pmacc::math::CT::max<SpeciesLowerMargin, FieldSolverLowerMargin>::type;

        for(uint32_t i=0; i<simDim; i++)
            {
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor SpeciesLowerMargin dim "<< i << " : " <<SpeciesLowerMargin().toRT()[i]<< std::endl;
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor FieldSolverLowerMargin dim "<< i << " : "<< FieldSolverLowerMargin().toRT()[i]<< std::endl;
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor LowerMargin dim "<< i << " : "<< LowerMargin().toRT()[i]<< std::endl;
            }


        /* ------------------ upper margin  -----------------------------------*/

        using SpeciesUpperMargin = pmacc::mp_fold<VectorSpeciesWithInterpolation,typename pmacc::math::CT::make_Int<simDim, 0>::type,SpeciesUpperMarginOp>;

        using FieldSolverUpperMargin = GetUpperMargin<fields::Solver>::type;

        using UpperMargin = pmacc::math::CT::max<SpeciesUpperMargin, FieldSolverUpperMargin>::type;

        for(uint32_t i=0; i<simDim; i++)
            {
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor SpeciesUpperMargin dim "<< i << " : " <<SpeciesUpperMargin().toRT()[i]<< std::endl;
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor FieldSolverUpperMargin dim "<< i << " : "<< FieldSolverUpperMargin().toRT()[i]<< std::endl;
                std::cout<< "Debug in picongpu/include/picongpu/fields/FieldRho.hpp/constructor UpperMargin dim "<< i << " : "<< UpperMargin().toRT()[i]<< std::endl;
            }


        const DataSpace<simDim> originGuard(LowerMargin().toRT());
        const DataSpace<simDim> endGuard(UpperMargin().toRT());
        //const DataSpace<simDim> originGuard{3,3,3};
        //const DataSpace<simDim> endGuard{3,3,3};
        //DataSpace<simDim> originGuard = GetLowerMargin<fields::Solver, FieldRho>::type::toRT();
        //DataSpace<simDim> endGuard = GetUpperMargin<fields::Solver, FieldRho>::type::toRT();

        /*go over all directions*/
        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim>(i);
            /*guarding cells depend on direction
            */
            DataSpace<simDim> guardingCells;
            for(uint32_t d = 0; d < simDim; ++d)
            {
                /*originGuard and endGuard are switch because we send data
                * e.g. from left I get endGuardingCells and from right I originGuardingCells
                */
                switch(relativMask[d])
                {
                    // receive from negativ side to positiv (end) guarding cells
                case -1:
                    guardingCells[d] = endGuard[d];
                    break;
                    // receive from positiv side to negativ (origin) guarding cells
                case 1:
                    guardingCells[d] = originGuard[d];
                    break;
                case 0:
                    guardingCells[d] = coreBorderSize[d];
                    break;
                };
            }

            buffer->addExchangeBuffer(i, guardingCells, m_commTagScatter);

            if(bufferRecv)
            {
                /* guarding cells depend on direction
                * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                * don't switch end and origin because this is a read buffer and not send buffer
                */
                for(uint32_t d = 0; d < simDim; ++d)
                    guardingCells[d] = (relativMask[d] == -1 ? originGuard[d] : endGuard[d]);
                bufferRecv->addExchange(GUARD, i, guardingCells, m_commTagGather);
            }
        }
    }

    //! Get a reference to the host-device buffer for the field values
    GridBuffer<FieldRho::ValueType, simDim>& FieldRho::getGridBuffer()
    {
        return *buffer;
    }

    GridBuffer<FieldRho::ValueType, simDim>* FieldRho::getGridBufferPointer()
    {
        return buffer.get();
    }

    //! Get the grid layout
    GridLayout<simDim> FieldRho::getGridLayout()
    {
        return cellDescription.getGridLayout();
    }

    //! Get the host data box for the field values
    FieldRho::DataBoxType FieldRho::getHostDataBox()
    {
        return buffer->getHostBuffer().getDataBox();
    }

    //! Get the device data box for the field values
    FieldRho::DataBoxType FieldRho::getDeviceDataBox()
    {
        return buffer->getDeviceBuffer().getDataBox();
    }


    void FieldRho::assign(FieldRho::ValueType value)
    {
        buffer->getDeviceBuffer().setValue(value);
    }


    /** Reset the host-device buffer for field values
     *
     * @param currentStep index of time iteration
     */
    void FieldRho::reset(uint32_t currentStep)
    {
        buffer->getHostBuffer().reset(true);
        buffer->getDeviceBuffer().reset(false);
    }

    //! Synchronize device data with host data
    void FieldRho::syncToDevice()
    {
        buffer->hostToDevice();
    }

    //! Synchronize host data with device data
    void FieldRho::synchronize()
    {
        buffer->deviceToHost();
    }


    //! Get unit of field components
    FieldRho::UnitValueType FieldRho::getUnit()
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

    std::vector<float_64> FieldRho::getUnitDimension()
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


    //! Get mapping for kernels
    MappingDesc FieldRho::getCellDescription()
    {
        return this->cellDescription;
    }

    //! Get text name
    std::string FieldRho::getName()
    {
        return "Rho";
    }

    //! Get id
    pmacc::SimulationDataId FieldRho::getUniqueId()
    {
        return id;
    }
    

    /** Start asynchronous send of field values
     *
     * Add data from the local guard of the GPU to the border of the neighboring GPUs.
     * This method can be called before or after asyncCommunicationGather without
     * explicit handling to avoid race conditions between both methods.
     *
     * @param serialEvent event to depend on
     */
    EventTask FieldRho::asyncCommunication(EventTask serialEvent)
    {
        EventTask ret;
        eventSystem::startTransaction(serialEvent + m_gatherEv + m_scatterEv);
        FieldFactory::getInstance().createTaskFieldReceiveAndInsert(*this);
        ret = eventSystem::endTransaction();

        eventSystem::startTransaction(serialEvent + m_gatherEv + m_scatterEv);
        FieldFactory::getInstance().createTaskFieldSend(*this);
        ret += eventSystem::endTransaction();
        m_scatterEv = ret;
        return ret;
    }

    /** Gather data from neighboring GPUs
     *
     * Copy data from the border of neighboring GPUs into the local guard.
     * This method can be called before or after asyncCommunication without
     * explicit handling to avoid race conditions between both methods.
     */
    EventTask FieldRho::asyncCommunicationGather(EventTask serialEvent)
    {
        PMACC_VERIFY_MSG(
            fieldTmpSupportGatherCommunication == true,
            "fieldTmpSupportGatherCommunication in memory.param must be set to true");

        if(bufferRecv != nullptr)
            m_gatherEv = bufferRecv->asyncCommunication(serialEvent + m_scatterEv + m_gatherEv);
        return m_gatherEv;
    }

    /** Bash particles in a direction.
     * Copy all particles from the guard of a direction to the device exchange buffer
     *
     * @param exchangeType exchange type
     */
    void FieldRho::bashField(uint32_t exchangeType)
    {
        pmacc::fields::operations::CopyGuardToExchange{}(*buffer, SuperCellSize{}, exchangeType);
    }

    /** Insert all particles which are in device exchange buffer
     *
     * @param exchangeType exchange type
     */
    void FieldRho::insertField(uint32_t exchangeType)
    {
        pmacc::fields::operations::AddExchangeToBorder{}(*buffer, SuperCellSize{}, exchangeType);
    }

} // namespace picongpu