/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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
 #include "picongpu/fields/FieldTmpOperations.hpp"
 #include "picongpu/fields/poissonSolver/FieldV.hpp"
 
 #include <pmacc/lockstep/lockstep.hpp>
 #include <pmacc/mappings/kernel/ExchangeMapping.hpp>
 #include <pmacc/memory/dataTypes/Mask.hpp>
 
 namespace picongpu::fields::poissonSolver
 {
     struct SolutionFunction
     {
#if 0
         HDINLINE auto operator()(math::Vector<double, simDim> const& totalCellCoordinate) const
         {
             if constexpr(simDim == 3u)
             {
                 return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y())
                        + 3.0 * math::sin(totalCellCoordinate.z())
                        + totalCellCoordinate.x() * totalCellCoordinate.productOfComponents() + 10.0;
             }
             else if constexpr(simDim == 2u)
             {
                 return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y())
                        + totalCellCoordinate.x() * totalCellCoordinate.productOfComponents() + 10.0;
             }
         }
#endif 
        HDINLINE auto operator()(math::Vector<double, simDim> const& totalCellCoordinate) const
        {
            if constexpr(simDim == 3u)
            {
                return 5.0 * math::sin(totalCellCoordinate.x()) + 2.0 * math::cos(totalCellCoordinate.y())
                    + 3.0 * math::sin(totalCellCoordinate.z()) + totalCellCoordinate.x() +  5.0 ;
            }
            else if constexpr(simDim == 2u)
            {
                return 5.0 * math::sin(totalCellCoordinate.x()) + 2.0 * math::cos(totalCellCoordinate.y())
                + totalCellCoordinate.x() +  5.0;
            }
        }
     };

     // RHSFunction = -Laplacian(SolutionFunction)
     struct RHSFunction
     {
#if 0
         HDINLINE auto operator()(math::Vector<double, simDim> const& totalCellCoordinate) const
         {
             if constexpr(simDim == 3u)
             {
                 return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y())
                        + 3.0 * math::sin(totalCellCoordinate.z()) - 2 * totalCellCoordinate.y() * totalCellCoordinate.z();
             }
             else if constexpr(simDim == 2u)
             {
                 return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y()) - 2 * totalCellCoordinate.y();
             }
         }
#endif
        HDINLINE auto operator()(math::Vector<double, simDim> const& totalCellCoordinate) const
        {
            if constexpr(simDim == 3u)
            {
                return 5.0 * math::sin(totalCellCoordinate.x()) + 2.0 * math::cos(totalCellCoordinate.y())
                    + 3.0 * math::sin(totalCellCoordinate.z());
            }
            else if constexpr(simDim == 2u)
            {
                return 5.0 * math::sin(totalCellCoordinate.x()) + 2.0 * math::cos(totalCellCoordinate.y());
            }
        }
     };
 
     struct SetSyntheticRHSKernel
     {
         DINLINE auto operator()(
             auto const& worker,
             auto fieldBox,
             auto const rhsFunction,
             DataSpace<simDim> cellOffsetToTotalOrigin,
             auto const mapper) const -> void
         {
             DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
             DataSpace<simDim> superCellTotalCellOffset
                 = cellOffsetToTotalOrigin + superCellIdx * SuperCellSize::toRT();
 
             constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;
 
             auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);
 
             forEachCellInSupercell(
                 [&](int32_t const linearCellIdx)
                 {
                     /* cell index within the superCell */
                     DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                     DataSpace<simDim> const totalCellIdx = superCellTotalCellOffset + cellIdx;
 
                     auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                          * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());
 
                    fieldBox(superCellIdx * SuperCellSize::toRT() + cellIdx) = rhsFunction(totalDistance);
                 });
         }
     };

     struct ComputeSyntheticErrorKernel
     {
         DINLINE auto operator()(
             auto const& worker,
             auto rBox,
             auto fieldVBox,
             auto const solutionFunction,
             DataSpace<simDim> cellOffsetToTotalOrigin,
             auto const mapper) const -> void
         {
             DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
             DataSpace<simDim> superCellTotalCellOffset
                 = cellOffsetToTotalOrigin + superCellIdx * SuperCellSize::toRT();
 
             constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;
 
             auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);
 
             forEachCellInSupercell(
                 [&](int32_t const linearCellIdx)
                 {
                     /* cell index within the superCell */
                     DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                     DataSpace<simDim> const totalCellIdx = superCellTotalCellOffset + cellIdx;
 
                     auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                          * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());
 
                    rBox(superCellIdx * SuperCellSize::toRT() + cellIdx) = solutionFunction(totalDistance) - 
                                                                            fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx);
                                                                            
                 });
         }
     };

     struct SetSyntheticRHS
     {
         void operator()(FieldTmp& fieldRho, MappingDesc mappingDesc) const
         {
             SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
             auto globalDomain = subGrid.getGlobalDomain();
             auto localDomain = subGrid.getLocalDomain();
 
             auto cellOffsetToTotalOrigin = globalDomain.offset + localDomain.offset;

             auto rhoBox = fieldRho.getDeviceDataBox();
             auto coreBorderMapper = makeAreaMapper<CORE + BORDER>(mappingDesc);
 
            PMACC_LOCKSTEP_KERNEL(SetSyntheticRHSKernel{})
                .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                    rhoBox,
                    RHSFunction{},
                    cellOffsetToTotalOrigin,
                    coreBorderMapper);
         }
     };

     struct ComputeSyntheticError
     {  
        template<typename T_buffer>
         void operator()(FieldV& fieldV, T_buffer& rBuffer, MappingDesc mappingDesc) const
         {
             SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
             auto globalDomain = subGrid.getGlobalDomain();
             auto localDomain = subGrid.getLocalDomain();
 
             auto cellOffsetToTotalOrigin = globalDomain.offset + localDomain.offset;

             auto fieldVBox = fieldV.fieldVBuffer->getDeviceBuffer().getDataBox();
             auto rBox = rBuffer.getDeviceBuffer().getDataBox();
             auto coreBorderMapper = makeAreaMapper<CORE + BORDER>(mappingDesc);
 
            PMACC_LOCKSTEP_KERNEL(ComputeSyntheticErrorKernel{})
                .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                    rBox,
                    fieldVBox,
                    SolutionFunction{},
                    cellOffsetToTotalOrigin,
                    coreBorderMapper);
         }
     };

 } // namespace picongpu::fields::poissonSolver
 