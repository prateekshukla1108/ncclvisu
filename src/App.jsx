import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Play, Pause, RotateCcw, Repeat, ChevronRight, ChevronLeft, Eye, EyeOff, BarChart3, Zap, Cpu, AlertTriangle, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

// === TYPES & CONSTANTS ===
const Operation = {
  AllReduce: 'AllReduce',
  Reduce: 'Reduce',
  Broadcast: 'Broadcast',
  AllGather: 'AllGather',
  ReduceScatter: 'ReduceScatter'
};

const Algorithm = {
  NVLS: 'NVLS (NVLink SHARP)',
  MultiShot: 'MultiShot (2-Shot)'
};

const Topology = {
  FourGPU: '4-GPU Example',
  SingleNode: 'Single Node (8x H100)',
  MultiNode: 'Multi Node (2x8 H100)'
};

// Algorithm compatibility matrix based on NCCL documentation
// NVLS uses SHARP ALUs for reduction OR multicast replication
// MultiShot = ReduceScatter + AllGather decomposition (only for symmetric ops)
const ALGORITHM_COMPATIBILITY = {
  [Operation.AllReduce]: [Algorithm.NVLS, Algorithm.MultiShot],
  [Operation.Reduce]: [Algorithm.NVLS],
  [Operation.Broadcast]: [Algorithm.NVLS],
  [Operation.AllGather]: [Algorithm.NVLS, Algorithm.MultiShot],
  [Operation.ReduceScatter]: [Algorithm.NVLS, Algorithm.MultiShot]
};

// Distinct colors for chunks
const CHUNK_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e',
  '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4889',
  '#f43f5e', '#84cc16', '#14b8a6', '#6366f1',
  '#a855f7', '#f59e0b', '#10b981', '#0ea5e9'
];

// === KNOWLEDGE DATA ===
// Mathematical and logical operation details for each operation/algorithm combination
const KNOWLEDGE_DATA = {
  operations: {
    [Operation.AllReduce]: {
      objective: 'Sum an array of size N across P GPUs so every GPU ends up with the full result.',
      math: {
        nvls: {
          phases: [
            { name: 'Push to Switch', operation: 'GPUs output data via multimem::red (Load Reduce) instruction' },
            { name: 'Switch Math', operation: 'NVSwitch v3 ALUs perform summation (Σ) as packets fly through crossbar' },
            { name: 'Broadcast', operation: 'Switch multicasts result to all GPUs via multimem::st' }
          ],
          complexity: 'O(1) latency (constant), 2 steps (Send + Receive)',
          advantage: 'GPUs don\'t waste SM cycles - "fire and forget"'
        }
      }
    },
    [Operation.Reduce]: {
      objective: 'Sum data from all GPUs onto a single root GPU.',
      math: {
        nvls: {
          phases: [
            { name: 'Push to Switch', operation: 'All GPUs send to multicast address' },
            { name: 'In-Fabric Reduce', operation: 'SHARP ALUs perform reduction' },
            { name: 'Unicast to Root', operation: 'Result delivered only to root GPU' }
          ],
          complexity: 'O(1) latency, native single-shot operation'
        }
      }
    },
    [Operation.Broadcast]: {
      objective: 'Distribute data from root GPU to all other GPUs.',
      math: {
        nvls: {
          phases: [
            { name: 'Root to Switch', operation: 'Root writes to multicast address' },
            { name: 'Physical Multicast', operation: 'Switch copies 1 input to N ports electrically' }
          ],
          complexity: 'L × 1 (Switch Latency) - instant replication'
        }
      }
    },
    [Operation.AllGather]: {
      objective: 'Gather N data from all GPUs onto every GPU (Output size: P × N).',
      math: {
        nvls: {
          phases: [
            { name: 'Send to Switch', operation: 'Each GPU writes chunk to multicast address' },
            { name: 'Multicast Amplify', operation: 'NVSwitch replicates all chunks to all GPUs' }
          ],
          note: 'NVLS wins on latency vs ring algorithms.'
        }
      }
    },
    [Operation.ReduceScatter]: {
      objective: 'Reduce data and scatter results so each GPU owns a unique reduced portion.',
      math: {
        nvls: {
          phases: [
            { name: 'Send to Switch', operation: 'GPUs issue multimem.ld_reduce' },
            { name: 'Per-Chunk Reduce', operation: 'SHARP ALUs perform reduction per chunk' },
            { name: 'Scatter Addressing', operation: 'Switch routes reduced chunk i to GPU i' }
          ],
          note: 'Critical for Tensor Parallelism workloads.'
        }
      }
    }
  },
  algorithms: {
    [Algorithm.NVLS]: {
      name: 'NVLink SHARP',
      description: 'In-network computing via NVSwitch v3 hardware.',
      hardware: 'SHARP ALUs: 400 GFlops FP32 compute inside switch',
      advantage: 'Offloads reduction to switch. GPUs do zero math.',
      bandwidth: '~480 GB/s AllReduce',
      complexity: 'O(1) latency - constant regardless of GPU count'
    },
    [Algorithm.MultiShot]: {
      name: '2-Shot Algorithm',
      description: 'Decomposes operation: ReduceScatter (Shot 1) + AllGather (Shot 2)',
      advantage: 'Both phases leverage NVLS primitives.',
      applicability: 'Only for symmetric operations (AllReduce, AllGather, ReduceScatter)',
      complexity: 'O(2) latency'
    }
  },
  hardware: {
    h100: {
      topology: {
        nodes: '8× H100 GPUs per node',
        interconnect: '4× NVSwitch v3 chips',
        bandwidth: '900 GB/s bidirectional per GPU (18 NVLinks)',
        connectivity: 'Fully Connected (Any-to-Any) - non-blocking fabric'
      },
      nvswitch: {
        feature: 'In-Network Computing',
        technology: 'SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)',
        capability: 'FP32 ALUs perform reduction mid-flight',
        benefit: 'Eliminates O(P) dependency → O(1) for GPUs'
      }
    },
    comparison: {
      headers: ['Feature', 'Legacy Ring', 'Hopper NVLS'],
      rows: [
        ['Steps', '2(P-1)', '2'],
        ['Latency', 'O(P) - Linear', 'O(1) - Constant'],
        ['Compute', 'GPU Cores', 'Switch ALUs (SHARP)'],
        ['Memory', 'Intermediate buffers', 'One-Shot (Direct Switch R/W)']
      ]
    }
  }
};

// clsx utility
const clsx = (...classes) => classes.filter(Boolean).join(' ');

// === SIMULATION ENGINE ===
const generateTimeline = (op, algo, topo) => {
  const NODE_COUNT = topo === Topology.FourGPU ? 4 : (topo === Topology.MultiNode ? 16 : 8);
  const CHUNK_COUNT = NODE_COUNT;

  // Use D0, D1, D2, D3 style labels for 4-GPU mode to match documentation
  const getDataLabel = (id) => topo === Topology.FourGPU ? `D${id}` : `C${id}`;
  const getGPULabel = (id) => topo === Topology.FourGPU ? `G${id}` : `G${id}`;

  const events = [];
  const steps = [];
  const bandwidthSamples = [];
  let currentTime = 0;

  const TRANSFER_TIME = 400;
  const INTER_NODE_MULTIPLIER = 1.5;
  const COMPUTE_TIME = 120;
  const SWITCH_COMPUTE_TIME = 80; // NVSwitch in-fabric reduction is fast
  const MULTICAST_TIME = 150; // Multicast is faster than sequential sends

  const bufferSnapshots = [];

  const initBufferState = () => {
    const state = {};
    for (let i = 0; i < NODE_COUNT; i++) {
      state[i] = Array.from({ length: CHUNK_COUNT }, (_, c) => ({
        chunkId: c,
        state: 'local',
        reductionCount: 1
      }));
    }
    return state;
  };

  let bufferState = initBufferState();

  const saveBufferSnapshot = () => {
    bufferSnapshots.push({
      time: currentTime,
      state: JSON.parse(JSON.stringify(bufferState))
    });
  };

  const isInterNode = (from, to) => {
    if (topo !== Topology.MultiNode) return false;
    return Math.floor(from / 8) !== Math.floor(to / 8);
  };

  const getTransferTime = (from, to) => {
    return isInterNode(from, to) ? TRANSFER_TIME * INTER_NODE_MULTIPLIER : TRANSFER_TIME;
  };

  const addStep = (desc, phase = '') => {
    saveBufferSnapshot();
    steps.push({ time: currentTime, description: desc, phase });
  };

  const addTransfer = (from, to, chunkId, label = '', direction = 'cw') => {
    const duration = getTransferTime(from, to);
    events.push({
      type: 'transfer',
      id: `t-${events.length}`,
      from,
      to,
      chunkId,
      color: CHUNK_COLORS[chunkId % CHUNK_COLORS.length],
      startTime: currentTime,
      duration,
      label: label || `C${chunkId}`,
      direction
    });

    const linkId = `${Math.min(from, to)}-${Math.max(from, to)}`;
    bandwidthSamples.push({
      time: currentTime,
      endTime: currentTime + duration,
      linkId,
      from,
      to,
      direction,
      utilization: 1.0
    });

    return duration;
  };

  const addCompute = (nodeId, label, duration = COMPUTE_TIME) => {
    events.push({
      type: 'compute',
      nodeId,
      label,
      startTime: currentTime,
      duration
    });
    return duration;
  };

  const addLinkActive = (from, to, duration, direction = 'cw') => {
    events.push({
      type: 'link',
      from,
      to,
      startTime: currentTime,
      duration,
      direction
    });
  };

  // In-switch reduction event
  const addSwitchReduce = (sources, chunkId, label = '') => {
    const duration = SWITCH_COMPUTE_TIME;
    events.push({
      type: 'switch-reduce',
      id: `sr-${events.length}`,
      sources,
      chunkId,
      color: CHUNK_COLORS[chunkId % CHUNK_COLORS.length],
      startTime: currentTime,
      duration,
      label: label || 'Σ'
    });
    return duration;
  };

  // === ALGORITHM IMPLEMENTATIONS ===


  // === NVLS (NVLink SHARP) ===
  if (algo === Algorithm.NVLS) {
    if (op === Operation.AllReduce) {
      addStep('NVLS AllReduce: Leverages NVSwitch in-network reduction via multicast objects', 'init');
      currentTime += 300;

      addStep('Step 1: All GPUs issue multimem.ld_reduce - sends local data TO switch for reduction', 'scatter');

      // All GPUs send to "switch" (multicast object) at once
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('Step 2: NVSwitch performs IN-FABRIC reduction (SHARP ALUs: 400 GFlops FP32)', 'switch-reduce');

      // Switch reduction event
      addSwitchReduce(Array.from({ length: NODE_COUNT }, (_, i) => i), 0, 'Σ');
      currentTime += SWITCH_COMPUTE_TIME;

      // Mark switch has reduced result
      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 'pending', state: 'in-switch', reductionCount: NODE_COUNT }];
      }
      saveBufferSnapshot();

      addStep('Step 3: Result available via multicast address - switch broadcasts to all GPUs', 'gather');

      // Switch multicasts result back to all GPUs
      events.push({
        type: 'from-switch',
        id: `fs-${events.length}`,
        destinations: Array.from({ length: NODE_COUNT }, (_, i) => i),
        chunkId: 0,
        color: '#22c55e',
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'Σ'
      });
      currentTime += MULTICAST_TIME;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 'all', state: 'final', reductionCount: NODE_COUNT }];
      }

      addStep('Complete! NVLS achieves ~480 GB/s AllReduce BW (vs ~370 GB/s Ring). SM offload!', 'done');
    }
    else if (op === Operation.ReduceScatter) {
      addStep('NVLS ReduceScatter: In-switch reduction with scattered output', 'init');
      currentTime += 300;

      addStep('All GPUs issue multimem.ld_reduce to send chunks to switch', 'scatter');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch SHARP ALUs perform reduction per chunk', 'switch-reduce');
      for (let c = 0; c < NODE_COUNT; c++) {
        addSwitchReduce(Array.from({ length: NODE_COUNT }, (_, i) => i), c, `Σ${c}`);
      }
      currentTime += SWITCH_COMPUTE_TIME;

      addStep('Switch routes reduced chunk i to GPU i (scatter addressing)', 'gather');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'from-switch-single',
          id: `fss-${events.length}`,
          destination: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME * 0.3,
          label: `Σ${gpu}`
        });
        bufferState[gpu] = [{ chunkId: gpu, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      currentTime += MULTICAST_TIME * 0.3;

      addStep('Complete! Each GPU owns fully reduced chunk i. Critical for Tensor Parallelism.', 'done');
    }
    else if (op === Operation.AllGather) {
      addStep('NVLS AllGather: NVSwitch multicast amplifies data to all GPUs', 'init');
      currentTime += 300;

      addStep('Each GPU issues multimem.st_multicast to write chunk to multicast address', 'scatter');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch replicates all chunks to all GPUs via multicast', 'broadcast');
      events.push({
        type: 'switch-broadcast',
        id: `sb-${events.length}`,
        destinations: Array.from({ length: NODE_COUNT }, (_, i) => i),
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'ALL'
      });
      currentTime += MULTICAST_TIME;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: 1
        }));
      }

      addStep('Complete! NOTE: Ring may achieve higher BW (~350 vs ~300 GB/s) due to TX/RX saturation. NVLS wins on latency.', 'done');
    }
    else if (op === Operation.Broadcast) {
      addStep('NVLS Broadcast: Single multimem.st_multicast from root → all GPUs receive', 'init');
      bufferState[0] = [{ chunkId: 0, state: 'source', reductionCount: 1 }];
      currentTime += 300;

      addStep('Root issues multimem.st_multicast (single NVLink transaction)', 'send');
      events.push({
        type: 'to-switch',
        id: `ts-${events.length}`,
        from: 0,
        chunkId: 0,
        color: CHUNK_COLORS[0],
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'Data'
      });
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch multicast replicates packet to all output ports simultaneously', 'multicast');
      events.push({
        type: 'from-switch',
        id: `fs-${events.length}`,
        destinations: Array.from({ length: NODE_COUNT }, (_, i) => i),
        chunkId: 0,
        color: CHUNK_COLORS[0],
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'Data'
      });
      currentTime += MULTICAST_TIME;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 0, state: 'final', reductionCount: 1 }];
      }

      addStep('Complete! True O(1) complexity broadcast via hardware multicast', 'done');
    }
    else if (op === Operation.Reduce) {
      addStep('NVLS Reduce: Native SHARP reduction - solves incast problem', 'init');
      currentTime += 300;

      addStep('All GPUs issue multimem.ld_reduce to send data to switch', 'scatter');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch SHARP ALUs perform in-fabric reduction', 'switch-reduce');
      addSwitchReduce(Array.from({ length: NODE_COUNT }, (_, i) => i), 0, 'Σ');
      currentTime += SWITCH_COMPUTE_TIME;

      addStep('Only root receives the single reduced result (no incast!)', 'gather');
      events.push({
        type: 'from-switch-single',
        id: `fss-${events.length}`,
        destination: 0,
        chunkId: 0,
        color: '#22c55e',
        startTime: currentTime,
        duration: MULTICAST_TIME * 0.3,
        label: 'Σ'
      });
      currentTime += MULTICAST_TIME * 0.3;

      bufferState[0] = [{ chunkId: 'all', state: 'reduced', reductionCount: NODE_COUNT }];
      for (let i = 1; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: i, state: 'sent', reductionCount: 1 }];
      }

      addStep('Complete! Incast eliminated - root receives 1 stream, not N-1. No MultiShot needed (already optimal).', 'done');
    }
    // NOTE: AllToAll with NVLS is NOT supported - NVLS is for reduction/replication, not routing
  }
  // === MULTISHOT (2-Shot AllReduce) ===
  else if (algo === Algorithm.MultiShot) {
    if (op === Operation.AllReduce) {
      addStep('MultiShot AllReduce: O(2) latency via NVLS primitives. Decomposed into ReduceScatter + AllGather.', 'init');
      currentTime += 300;

      addStep('SHOT 1 (ReduceScatter via NVLS): All GPUs issue multimem.ld_reduce for their chunks', 'shot1');

      // All GPUs send their data to switch - switch does the reduction!
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch SHARP ALUs perform IN-SWITCH reduction (not on GPUs!)', 'switch-reduce');

      // Switch performs reduction - THIS IS THE KEY DIFFERENCE
      for (let c = 0; c < NODE_COUNT; c++) {
        addSwitchReduce(Array.from({ length: NODE_COUNT }, (_, i) => i), c, `Σ${c}`);
      }
      currentTime += SWITCH_COMPUTE_TIME;

      // Reduced chunks scattered to owners
      addStep('Switch routes reduced chunk i to GPU i (ReduceScatter complete)', 'scatter-result');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'from-switch-single',
          id: `fss-${events.length}`,
          destination: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME * 0.5,
          label: `Σ${gpu}`
        });
        bufferState[gpu] = [{ chunkId: gpu, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      currentTime += MULTICAST_TIME * 0.5;
      saveBufferSnapshot();

      addStep('SHOT 2 (AllGather via NVLS): Each GPU issues multimem.st_multicast for its reduced chunk', 'shot2');

      // Each GPU multicasts its reduced chunk via switch
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts2-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME * 0.5,
          label: `Σ${gpu}`
        });
      }
      currentTime += MULTICAST_TIME * 0.5;

      addStep('NVSwitch multicasts all reduced chunks to all GPUs', 'broadcast');
      events.push({
        type: 'switch-broadcast',
        id: `sb-${events.length}`,
        destinations: Array.from({ length: NODE_COUNT }, (_, i) => i),
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'ALL'
      });
      currentTime += MULTICAST_TIME;

      // All GPUs now have all reduced chunks
      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: NODE_COUNT
        }));
      }

      addStep('Complete! 2 shots total. ~3× faster than Ring for small messages. SM offload achieved!', 'done');
    }
    else if (op === Operation.ReduceScatter) {
      addStep('MultiShot ReduceScatter: Single shot via NVLS (Phase 1 of MultiShot AllReduce)', 'init');
      currentTime += 300;

      addStep('All GPUs issue multimem.ld_reduce to send data to switch', 'scatter');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch SHARP ALUs perform in-switch reduction', 'switch-reduce');
      for (let c = 0; c < NODE_COUNT; c++) {
        addSwitchReduce(Array.from({ length: NODE_COUNT }, (_, i) => i), c, `Σ${c}`);
      }
      currentTime += SWITCH_COMPUTE_TIME;

      addStep('Switch routes reduced chunk i to GPU i', 'scatter-result');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'from-switch-single',
          id: `fss-${events.length}`,
          destination: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME * 0.3,
          label: `Σ${gpu}`
        });
        bufferState[gpu] = [{ chunkId: gpu, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      currentTime += MULTICAST_TIME * 0.3;

      addStep('Complete! Single communication step via NVLS. Critical for TP efficiency.', 'done');
    }
    else if (op === Operation.AllGather) {
      addStep('MultiShot AllGather: Single shot via NVLS multicast (Phase 2 of MultiShot AllReduce)', 'init');
      currentTime += 300;

      addStep('All GPUs issue multimem.st_multicast for their chunks', 'multicast');
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        events.push({
          type: 'to-switch',
          id: `ts-${events.length}`,
          from: gpu,
          chunkId: gpu,
          color: CHUNK_COLORS[gpu % CHUNK_COLORS.length],
          startTime: currentTime,
          duration: MULTICAST_TIME,
          label: getDataLabel(gpu)
        });
      }
      currentTime += MULTICAST_TIME;

      addStep('NVSwitch multicasts all chunks to all GPUs simultaneously', 'broadcast');
      events.push({
        type: 'switch-broadcast',
        id: `sb-${events.length}`,
        destinations: Array.from({ length: NODE_COUNT }, (_, i) => i),
        startTime: currentTime,
        duration: MULTICAST_TIME,
        label: 'ALL'
      });
      currentTime += MULTICAST_TIME;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: 1
        }));
      }

      addStep('Complete! Single step via switch multicast amplification', 'done');
    }
    // NOTE: MultiShot Reduce is NOT a thing - NVLS Reduce is already single-shot optimal
    // NOTE: MultiShot AllToAll is NOT a thing - cannot be decomposed into RS+AG
  }
  // === RING ALGORITHMS ===


  saveBufferSnapshot();

  return {
    duration: currentTime + 500,
    events,
    steps,
    bufferSnapshots,
    bandwidthSamples,
    nodeCount: NODE_COUNT
  };
};

// === NODE LAYOUT ===
const getNodes = (topo) => {
  const nodes = [];
  const links = [];

  if (topo === Topology.FourGPU) {
    // 4-GPU layout matching documentation examples
    const radius = 160;
    const cx = 400, cy = 300;

    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2 - Math.PI / 2;
      nodes.push({
        id: i,
        label: `GPU ${i}`,
        dataLabel: `D${i}`,  // Documentation-style data label
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius
      });

      // Ring connections for educational visualization
      links.push({ from: i, to: (i + 1) % 4, type: 'ring' });

      // Mesh connections (all-to-all via NVSwitch)
      for (let j = i + 2; j < 4; j++) {
        if (j !== (i + 3) % 4) {
          links.push({ from: i, to: j, type: 'mesh' });
        }
      }
    }
  } else if (topo === Topology.SingleNode) {
    const radius = 180;
    const cx = 400, cy = 300;

    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
      nodes.push({
        id: i,
        label: `GPU ${i}`,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius
      });

      links.push({ from: i, to: (i + 1) % 8, type: 'ring' });

      for (let j = i + 2; j < 8; j++) {
        if (j !== (i + 7) % 8) {
          links.push({ from: i, to: j, type: 'mesh' });
        }
      }
    }
  } else {
    const radius = 110;
    const cx1 = 200, cy1 = 300;
    const cx2 = 600, cy2 = 300;

    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
      nodes.push({
        id: i,
        label: `N0:G${i}`,
        x: cx1 + Math.cos(angle) * radius,
        y: cy1 + Math.sin(angle) * radius,
        nodeId: 0
      });

      links.push({ from: i, to: (i + 1) % 8, type: 'ring' });
      for (let j = i + 2; j < 8; j++) {
        if (j !== (i + 7) % 8) {
          links.push({ from: i, to: j, type: 'mesh' });
        }
      }
    }

    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
      const gpuId = i + 8;
      nodes.push({
        id: gpuId,
        label: `N1:G${i}`,
        x: cx2 + Math.cos(angle) * radius,
        y: cy2 + Math.sin(angle) * radius,
        nodeId: 1
      });

      links.push({ from: gpuId, to: 8 + (i + 1) % 8, type: 'ring' });
      for (let j = i + 2; j < 8; j++) {
        if (j !== (i + 7) % 8) {
          links.push({ from: gpuId, to: 8 + j, type: 'mesh' });
        }
      }
    }

    links.push({ from: 7, to: 8, type: 'inter' });
    links.push({ from: 15, to: 0, type: 'inter' });

    for (let i = 1; i < 7; i++) {
      links.push({ from: i, to: i + 8, type: 'inter-mesh' });
    }
  }

  return { nodes, links };
};

// === BANDWIDTH CHART COMPONENT ===
const BandwidthChart = ({ timeline, currentTime, topology }) => {
  if (!timeline) return null;

  const width = 280;
  const height = 100;
  const padding = { top: 20, right: 10, bottom: 25, left: 35 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const bucketCount = 50;
  const bucketDuration = timeline.duration / bucketCount;

  const utilizationData = useMemo(() => {
    const buckets = Array(bucketCount).fill(0);
    const maxLinks = topology === Topology.MultiNode ? 16 : 8;

    timeline.bandwidthSamples.forEach(sample => {
      const startBucket = Math.floor(sample.time / bucketDuration);
      const endBucket = Math.floor(sample.endTime / bucketDuration);

      for (let b = startBucket; b <= endBucket && b < bucketCount; b++) {
        buckets[b] += 1 / maxLinks;
      }
    });

    return buckets.map((v, i) => ({
      time: i * bucketDuration,
      utilization: Math.min(1, v)
    }));
  }, [timeline, bucketDuration, topology]);

  const maxUtil = Math.max(...utilizationData.map(d => d.utilization), 0.1);

  return (
    <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
      <div className="text-xs text-slate-400 mb-2 flex items-center gap-2">
        <BarChart3 size={14} />
        <span>Bandwidth Utilization</span>
      </div>
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id="utilGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#22c55e" stopOpacity="0.1" />
          </linearGradient>
        </defs>

        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {[0, 0.5, 1].map(v => (
            <g key={v}>
              <line
                x1={0} y1={chartHeight * (1 - v)}
                x2={chartWidth} y2={chartHeight * (1 - v)}
                className="stroke-slate-700"
                strokeDasharray="2 2"
              />
              <text
                x={-5} y={chartHeight * (1 - v) + 3}
                className="fill-slate-500 text-[8px]"
                textAnchor="end"
              >
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          <path
            d={`M 0 ${chartHeight} ${utilizationData.map((d, i) =>
              `L ${(i / bucketCount) * chartWidth} ${chartHeight * (1 - d.utilization / maxUtil)}`
            ).join(' ')} L ${chartWidth} ${chartHeight} Z`}
            fill="url(#utilGradient)"
          />

          <path
            d={`M ${utilizationData.map((d, i) =>
              `${(i / bucketCount) * chartWidth} ${chartHeight * (1 - d.utilization / maxUtil)}`
            ).join(' L ')}`}
            fill="none"
            className="stroke-green-400"
            strokeWidth={1.5}
          />

          <line
            x1={(currentTime / timeline.duration) * chartWidth}
            y1={0}
            x2={(currentTime / timeline.duration) * chartWidth}
            y2={chartHeight}
            className="stroke-white"
            strokeWidth={1}
          />
        </g>

        <text x={padding.left} y={height - 5} className="fill-slate-500 text-[8px]">0</text>
        <text x={width - padding.right} y={height - 5} className="fill-slate-500 text-[8px]" textAnchor="end">
          {(timeline.duration / 1000).toFixed(1)}s
        </text>
      </svg>
    </div>
  );
};

// === BUFFER STATE COMPONENT ===
const BufferStateView = ({ timeline, currentTime, nodeCount }) => {
  if (!timeline || !timeline.bufferSnapshots.length) return null;

  let currentSnapshot = timeline.bufferSnapshots[0];
  for (const snapshot of timeline.bufferSnapshots) {
    if (snapshot.time <= currentTime) {
      currentSnapshot = snapshot;
    } else {
      break;
    }
  }

  const state = currentSnapshot.state;
  const gpuIds = Object.keys(state).map(Number).sort((a, b) => a - b);

  return (
    <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
      <div className="text-xs text-slate-400 mb-2 flex items-center gap-2">
        <Eye size={14} />
        <span>GPU Buffer State</span>
      </div>
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {gpuIds.slice(0, 8).map(gpuId => {
          const chunks = state[gpuId] || [];
          return (
            <div key={gpuId} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-500 w-10 flex-shrink-0">
                GPU{gpuId}
              </span>
              <div className="flex gap-0.5 flex-wrap">
                {chunks.slice(0, 8).map((chunk, i) => {
                  const isReduced = chunk.state === 'reduced' || chunk.state === 'final';
                  const isPartial = chunk.state === 'partial';
                  const isSent = chunk.state === 'sent';
                  const isInSwitch = chunk.state === 'in-switch';
                  const color = typeof chunk.chunkId === 'number'
                    ? CHUNK_COLORS[chunk.chunkId % CHUNK_COLORS.length]
                    : '#22c55e';

                  return (
                    <div
                      key={i}
                      className={clsx(
                        "w-4 h-4 rounded-sm flex items-center justify-center text-[7px] font-bold",
                        isReduced ? "ring-1 ring-white/50" : "",
                        isPartial ? "opacity-60" : "",
                        isSent ? "opacity-30" : "",
                        isInSwitch ? "animate-pulse" : ""
                      )}
                      style={{ backgroundColor: isInSwitch ? '#8b5cf6' : color }}
                      title={`Chunk ${chunk.chunkId}: ${chunk.state} (${chunk.reductionCount || 1}/${nodeCount} reduced)`}
                    >
                      {typeof chunk.chunkId === 'number' ? chunk.chunkId : 'Σ'}
                    </div>
                  );
                })}
                {chunks.length > 8 && (
                  <span className="text-[8px] text-slate-500">+{chunks.length - 8}</span>
                )}
              </div>
            </div>
          );
        })}
        {gpuIds.length > 8 && (
          <div className="text-[9px] text-slate-500 text-center pt-1">
            ... and {gpuIds.length - 8} more GPUs
          </div>
        )}
      </div>
      <div className="flex gap-3 mt-2 pt-2 border-t border-slate-700 text-[8px] text-slate-500">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-slate-600"></div>
          <span>Local</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-slate-600 opacity-60"></div>
          <span>Partial</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-violet-500 animate-pulse"></div>
          <span>In Switch</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-green-500 ring-1 ring-white/50"></div>
          <span>Final</span>
        </div>
      </div>
    </div>
  );
};

// === KNOWLEDGE PANEL COMPONENT ===
const KnowledgePanel = ({ operation, algorithm }) => {
  const [expandedSection, setExpandedSection] = useState('operation');

  const opData = KNOWLEDGE_DATA.operations[operation];
  const algoData = KNOWLEDGE_DATA.algorithms[algorithm];
  const isNVLS = algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot;

  const getMathData = () => {
    if (!opData?.math) return null;
    if (operation === Operation.AllToAll) return opData.math;
    if (isNVLS && opData.math.nvls) return opData.math.nvls;
    if (opData.math.ring) return opData.math.ring;
    return null;
  };

  const mathData = getMathData();

  const toggleSection = (section) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
      <div className="text-xs text-slate-400 p-3 pb-2 flex items-center gap-2 border-b border-slate-700/50">
        <BookOpen size={14} />
        <span className="font-medium">Knowledge Base</span>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {/* Operation Section */}
        <div className="border-b border-slate-700/50">
          <button
            onClick={() => toggleSection('operation')}
            className="w-full p-2.5 flex items-center justify-between text-left hover:bg-slate-700/30 transition-colors"
          >
            <span className="text-[10px] font-semibold text-green-400 uppercase tracking-wider">
              {operation}
            </span>
            {expandedSection === 'operation' ? <ChevronUp size={12} className="text-slate-500" /> : <ChevronDown size={12} className="text-slate-500" />}
          </button>

          {expandedSection === 'operation' && opData && (
            <div className="px-3 pb-3 space-y-2">
              <div className="text-[10px] text-slate-300 leading-relaxed">
                <span className="text-slate-500">Objective:</span> {opData.objective}
              </div>

              {operation === Operation.AllToAll && opData.math && (
                <div className="space-y-1.5">
                  <div className="text-[9px] text-slate-400">
                    <span className="text-amber-400/80">Logic:</span> {opData.math.logic}
                  </div>
                  <div className="text-[9px] text-slate-400">
                    <span className="text-cyan-400/80">Data:</span> {opData.math.dataMovement}
                  </div>
                  <div className="text-[9px] text-slate-400">
                    <span className="text-blue-400/80">Complexity:</span> {opData.math.complexity}
                  </div>
                  {opData.math.note && (
                    <div className="text-[8px] text-amber-300/70 bg-amber-900/20 p-1.5 rounded border border-amber-700/30">
                      {opData.math.note}
                    </div>
                  )}
                </div>
              )}

              {mathData?.phases && (
                <div className="space-y-1.5">
                  <div className="text-[9px] text-slate-500 uppercase tracking-wider">Phases</div>
                  {mathData.phases.map((phase, i) => (
                    <div key={i} className={clsx(
                      "p-2 rounded text-[9px] border",
                      isNVLS ? "bg-violet-900/20 border-violet-700/30" : "bg-slate-700/30 border-slate-600/30"
                    )}>
                      <div className={clsx(
                        "font-semibold mb-0.5",
                        isNVLS ? "text-violet-300" : "text-blue-300"
                      )}>
                        {phase.name} {phase.steps && <span className="text-slate-500 font-normal">({phase.steps})</span>}
                      </div>
                      <div className="text-slate-400">{phase.operation}</div>
                      {phase.result && <div className="text-slate-500 mt-0.5">→ {phase.result}</div>}
                      {phase.pattern && <div className="text-slate-500 mt-0.5 font-mono">{phase.pattern}</div>}
                      {phase.volume && <div className="text-slate-500 mt-0.5">{phase.volume}</div>}
                    </div>
                  ))}
                </div>
              )}

              {mathData?.formula && (
                <div className="text-[9px] text-cyan-400/80 font-mono bg-slate-900/50 p-1.5 rounded">
                  {mathData.formula}
                </div>
              )}

              {mathData?.totalData && (
                <div className="text-[9px] text-slate-400">
                  <span className="text-slate-500">Total Data:</span> <span className="font-mono">{mathData.totalData}</span>
                </div>
              )}

              {mathData?.complexity && (
                <div className={clsx(
                  "text-[9px] p-1.5 rounded",
                  isNVLS ? "bg-violet-900/30 text-violet-300" : "bg-blue-900/30 text-blue-300"
                )}>
                  <span className="text-slate-400">Complexity:</span> {mathData.complexity}
                </div>
              )}

              {mathData?.advantage && (
                <div className="text-[9px] text-green-400/80 bg-green-900/20 p-1.5 rounded border border-green-700/30">
                  {mathData.advantage}
                </div>
              )}

              {mathData?.note && (
                <div className="text-[8px] text-slate-400 italic">
                  {mathData.note}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Algorithm Section */}
        <div className="border-b border-slate-700/50">
          <button
            onClick={() => toggleSection('algorithm')}
            className="w-full p-2.5 flex items-center justify-between text-left hover:bg-slate-700/30 transition-colors"
          >
            <span className={clsx(
              "text-[10px] font-semibold uppercase tracking-wider",
              isNVLS ? "text-violet-400" : "text-blue-400"
            )}>
              {algoData?.name || algorithm}
            </span>
            {expandedSection === 'algorithm' ? <ChevronUp size={12} className="text-slate-500" /> : <ChevronDown size={12} className="text-slate-500" />}
          </button>

          {expandedSection === 'algorithm' && algoData && (
            <div className="px-3 pb-3 space-y-2">
              <div className="text-[10px] text-slate-300 leading-relaxed">
                {algoData.description}
              </div>

              {algoData.pattern && (
                <div className="text-[9px] font-mono text-amber-400/80 bg-amber-900/20 p-1.5 rounded">
                  {algoData.pattern}
                </div>
              )}

              {algoData.advantage && (
                <div className="text-[9px] text-green-400/80">
                  <span className="text-slate-500">Advantage:</span> {algoData.advantage}
                </div>
              )}

              {algoData.weakness && (
                <div className="text-[9px] text-red-400/80">
                  <span className="text-slate-500">Weakness:</span> {algoData.weakness}
                </div>
              )}

              {algoData.steps && (
                <div className="text-[9px] text-slate-400">
                  <span className="text-slate-500">Steps:</span> <span className="font-mono">{algoData.steps}</span>
                </div>
              )}

              {algoData.bandwidth && (
                <div className="text-[9px] text-cyan-400/80">
                  <span className="text-slate-500">Bandwidth:</span> {algoData.bandwidth}
                </div>
              )}

              {algoData.hardware && (
                <div className="text-[9px] text-violet-300 bg-violet-900/20 p-1.5 rounded border border-violet-700/30">
                  <Cpu size={10} className="inline mr-1" />
                  {algoData.hardware}
                </div>
              )}

              {algoData.complexity && (
                <div className={clsx(
                  "text-[9px] font-semibold",
                  isNVLS ? "text-violet-300" : "text-blue-300"
                )}>
                  {algoData.complexity}
                </div>
              )}

              {algoData.applicability && (
                <div className="text-[8px] text-slate-500 italic">
                  {algoData.applicability}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Hardware Comparison Section */}
        <div>
          <button
            onClick={() => toggleSection('hardware')}
            className="w-full p-2.5 flex items-center justify-between text-left hover:bg-slate-700/30 transition-colors"
          >
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Zap size={10} className="text-violet-400" />
              H100 Comparison
            </span>
            {expandedSection === 'hardware' ? <ChevronUp size={12} className="text-slate-500" /> : <ChevronDown size={12} className="text-slate-500" />}
          </button>

          {expandedSection === 'hardware' && (
            <div className="px-3 pb-3 space-y-2">
              {/* Comparison Table */}
              <div className="overflow-hidden rounded border border-slate-600/50">
                <table className="w-full text-[8px]">
                  <thead>
                    <tr className="bg-slate-700/50">
                      {KNOWLEDGE_DATA.hardware.comparison.headers.map((h, i) => (
                        <th key={i} className={clsx(
                          "p-1.5 text-left font-semibold",
                          i === 0 ? "text-slate-400" : i === 1 ? "text-blue-400" : "text-violet-400"
                        )}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {KNOWLEDGE_DATA.hardware.comparison.rows.map((row, i) => (
                      <tr key={i} className="border-t border-slate-700/50">
                        {row.map((cell, j) => (
                          <td key={j} className={clsx(
                            "p-1.5",
                            j === 0 ? "text-slate-400 font-medium" : "text-slate-300 font-mono"
                          )}>
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Hardware Details */}
              <div className="space-y-1.5">
                <div className="text-[9px] text-slate-400">
                  <span className="text-slate-500">Topology:</span> {KNOWLEDGE_DATA.hardware.h100.topology.nodes}
                </div>
                <div className="text-[9px] text-slate-400">
                  <span className="text-slate-500">Interconnect:</span> {KNOWLEDGE_DATA.hardware.h100.topology.interconnect}
                </div>
                <div className="text-[9px] text-cyan-400/80">
                  <span className="text-slate-500">Bandwidth:</span> {KNOWLEDGE_DATA.hardware.h100.topology.bandwidth}
                </div>
                <div className="text-[9px] text-slate-400">
                  <span className="text-slate-500">Connectivity:</span> {KNOWLEDGE_DATA.hardware.h100.topology.connectivity}
                </div>
              </div>

              {/* NVSwitch Details */}
              <div className="bg-violet-900/20 p-2 rounded border border-violet-700/30 space-y-1">
                <div className="text-[9px] text-violet-300 font-semibold flex items-center gap-1">
                  <Zap size={10} />
                  NVSwitch v3 SHARP
                </div>
                <div className="text-[8px] text-slate-400">
                  {KNOWLEDGE_DATA.hardware.h100.nvswitch.technology}
                </div>
                <div className="text-[8px] text-slate-400">
                  {KNOWLEDGE_DATA.hardware.h100.nvswitch.capability}
                </div>
                <div className="text-[8px] text-green-400/80">
                  {KNOWLEDGE_DATA.hardware.h100.nvswitch.benefit}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// === MAIN COMPONENT ===
const App = () => {
  const [operation, setOperation] = useState(Operation.AllReduce);
  const [algorithm, setAlgorithm] = useState(Algorithm.NVLS);
  const [topology, setTopology] = useState(Topology.SingleNode);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [timeline, setTimeline] = useState(null);
  const [showBuffers, setShowBuffers] = useState(true);
  const [showBandwidth, setShowBandwidth] = useState(true);
  const [showKnowledge, setShowKnowledge] = useState(true);

  const lastTimeRef = useRef(0);

  // Get compatible algorithms for current operation
  const compatibleAlgorithms = ALGORITHM_COMPATIBILITY[operation] || [];

  // Auto-select compatible algorithm when operation changes
  useEffect(() => {
    if (!compatibleAlgorithms.includes(algorithm)) {
      setAlgorithm(compatibleAlgorithms[0] || Algorithm.NVLS);
    }
  }, [operation, compatibleAlgorithms, algorithm]);

  useEffect(() => {
    // Only generate timeline if algorithm is compatible
    if (compatibleAlgorithms.includes(algorithm)) {
      const tl = generateTimeline(operation, algorithm, topology);
      setTimeline(tl);
      setProgress(0);
      setIsPlaying(false);
    }
  }, [operation, algorithm, topology, compatibleAlgorithms]);

  useEffect(() => {
    let animationFrameId;

    const animate = (time) => {
      if (isPlaying && timeline) {
        if (!lastTimeRef.current) lastTimeRef.current = time;
        const delta = time - lastTimeRef.current;
        lastTimeRef.current = time;

        const deltaSeconds = delta / 1000;
        const totalDurationSeconds = timeline.duration / 1000;

        setProgress(p => {
          const newP = p + (deltaSeconds * speed) / totalDurationSeconds;
          if (newP >= 1) {
            if (isLooping) {
              lastTimeRef.current = time;
              return 0;
            }
            setIsPlaying(false);
            return 1;
          }
          return newP;
        });
      } else {
        lastTimeRef.current = 0;
      }
      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, timeline, speed, isLooping]);

  const currentTime = timeline ? progress * timeline.duration : 0;
  const { nodes, links } = useMemo(() => getNodes(topology), [topology]);

  // Active events calculation
  const activePackets = useMemo(() => {
    if (!timeline) return [];
    return timeline.events
      .filter(e => e.type === 'transfer' || e.type === 'parallel-transfer')
      .filter(e => currentTime >= e.startTime && currentTime <= e.startTime + e.duration)
      .map(e => {
        const t = Math.min(1, Math.max(0, (currentTime - e.startTime) / e.duration));
        return { ...e, t };
      });
  }, [timeline, currentTime]);

  const activeSwitchEvents = useMemo(() => {
    if (!timeline) return [];
    return timeline.events
      .filter(e => ['to-switch', 'from-switch', 'from-switch-single', 'switch-reduce',
        'switch-broadcast', 'multicast', 'multicast-out', 'to-switch-multi',
        'switch-route', 'from-switch-gather'].includes(e.type))
      .filter(e => currentTime >= e.startTime && currentTime <= e.startTime + e.duration)
      .map(e => {
        const t = Math.min(1, Math.max(0, (currentTime - e.startTime) / e.duration));
        return { ...e, t };
      });
  }, [timeline, currentTime]);

  const activeComputes = useMemo(() => {
    if (!timeline) return [];
    return timeline.events
      .filter(e => e.type === 'compute')
      .filter(e => currentTime >= e.startTime && currentTime <= e.startTime + e.duration);
  }, [timeline, currentTime]);

  const activeLinks = useMemo(() => {
    if (!timeline) return new Map();
    const active = new Map();
    timeline.events
      .filter(e => e.type === 'link')
      .filter(e => currentTime >= e.startTime && currentTime <= e.startTime + e.duration)
      .forEach(e => {
        const key = `${e.from}-${e.to}`;
        active.set(key, e.direction || 'cw');
      });
    return active;
  }, [timeline, currentTime]);

  const currentStep = useMemo(() => {
    if (!timeline || !timeline.steps.length) return null;
    let step = timeline.steps[0];
    for (const s of timeline.steps) {
      if (s.time <= currentTime) step = s;
      else break;
    }
    return step;
  }, [timeline, currentTime]);

  const togglePlay = () => setIsPlaying(!isPlaying);
  const toggleLoop = () => setIsLooping(!isLooping);
  const reset = () => { setIsPlaying(false); setProgress(0); };

  const stepForward = () => {
    if (!timeline) return;
    const idx = timeline.steps.findIndex((s, i) =>
      i === timeline.steps.length - 1 || timeline.steps[i + 1].time > currentTime
    );
    if (idx < timeline.steps.length - 1) {
      setProgress(timeline.steps[idx + 1].time / timeline.duration);
    }
  };

  const stepBackward = () => {
    if (!timeline) return;
    const idx = timeline.steps.findIndex((s, i) =>
      i === timeline.steps.length - 1 || timeline.steps[i + 1].time > currentTime
    );
    if (idx > 0) {
      setProgress(timeline.steps[idx - 1].time / timeline.duration);
    }
  };

  const width = 800;
  const height = 550;

  // NVSwitch position
  const switchX = topology === Topology.MultiNode ? 400 : 400;
  const switchY = 300;

  // Check if NVLS/MultiShot algorithm is active
  const isNVLSAlgo = algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot;
  const hasSwitchActivity = activeSwitchEvents.length > 0;

  return (
    <div className="flex h-screen bg-slate-950 text-white font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 flex-shrink-0 border-r border-slate-800 bg-slate-900/50 p-4 flex flex-col gap-3 overflow-y-auto">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-7 h-7 rounded bg-gradient-to-tr from-green-500 to-emerald-400 flex items-center justify-center font-bold text-slate-900 text-sm">H</div>
          <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-400">Hopper Visualizer</h1>
          <span className="text-[9px] px-1.5 py-0.5 bg-violet-500/20 text-violet-300 rounded border border-violet-500/30">H100 ONLY</span>
        </div>

        {/* Config */}
        <div className="space-y-1.5">
          <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Topology</h3>
          <select
            className="w-full bg-slate-800 border border-slate-700 rounded p-1.5 text-sm focus:ring-2 focus:ring-green-500 outline-none"
            value={topology}
            onChange={(e) => setTopology(e.target.value)}
          >
            {Object.values(Topology).map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div className="space-y-1.5">
          <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Operation</h3>
          <div className="grid grid-cols-2 gap-1.5">
            {Object.values(Operation).map(op => (
              <button
                key={op}
                onClick={() => setOperation(op)}
                className={clsx(
                  "p-1.5 text-xs rounded border transition-all",
                  operation === op
                    ? "bg-green-500/20 border-green-500 text-green-400 font-medium"
                    : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                )}
              >
                {op}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1.5">
          <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
            Algorithm
            <span className="text-[8px] px-1 py-0.5 bg-violet-500/30 text-violet-300 rounded flex items-center gap-1">
              <Zap size={8} /> Hopper
            </span>
          </h3>
          <div className="flex flex-col gap-1.5">
            {compatibleAlgorithms.map(algo => {
              const isSelected = algorithm === algo;

              return (
                <button
                  key={algo}
                  onClick={() => setAlgorithm(algo)}
                  className={clsx(
                    "p-2 text-xs rounded border text-left transition-all flex items-center gap-2",
                    isSelected
                      ? "bg-violet-500/20 border-violet-500 text-violet-300"
                      : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                  )}
                >
                  <div className={clsx(
                    "w-1.5 h-1.5 rounded-full",
                    isSelected ? "bg-violet-400" : "bg-slate-600"
                  )} />
                  <span className="flex-1">{algo}</span>
                  <Cpu size={12} className="text-violet-400" />
                </button>
              );
            })}
          </div>

          {operation === Operation.Reduce && (
            <div className="mt-2 p-2 bg-violet-900/20 border border-violet-700/30 rounded text-[9px] text-violet-300/80">
              <div className="flex items-start gap-1.5">
                <Zap size={10} className="mt-0.5 flex-shrink-0" />
                <span>
                  <strong>Reduce</strong> is natively optimal via NVLS (single-shot).
                  MultiShot is only for symmetric operations.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Current Step */}
        {currentStep && (
          <div className={clsx(
            "rounded-lg p-2.5 border",
            (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
              ? "bg-violet-900/30 border-violet-700"
              : "bg-slate-800/50 border-slate-700"
          )}>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Status</div>
            <div className="text-xs text-slate-300 leading-relaxed">{currentStep.description}</div>
          </div>
        )}

        {/* Analysis Panels */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-slate-500 uppercase">Analysis</span>
            <div className="flex gap-1">
              <button
                onClick={() => setShowBandwidth(!showBandwidth)}
                className={clsx("p-1 rounded", showBandwidth ? "text-green-400" : "text-slate-600")}
              >
                <BarChart3 size={14} />
              </button>
              <button
                onClick={() => setShowBuffers(!showBuffers)}
                className={clsx("p-1 rounded", showBuffers ? "text-green-400" : "text-slate-600")}
              >
                {showBuffers ? <Eye size={14} /> : <EyeOff size={14} />}
              </button>
              <button
                onClick={() => setShowKnowledge(!showKnowledge)}
                className={clsx("p-1 rounded", showKnowledge ? "text-violet-400" : "text-slate-600")}
                title="Knowledge Base"
              >
                <BookOpen size={14} />
              </button>
            </div>
          </div>

          {showBandwidth && (
            <BandwidthChart
              timeline={timeline}
              currentTime={currentTime}
              topology={topology}
            />
          )}

          {showBuffers && (
            <BufferStateView
              timeline={timeline}
              currentTime={currentTime}
              nodeCount={timeline?.nodeCount || 8}
            />
          )}

          {showKnowledge && (
            <KnowledgePanel
              operation={operation}
              algorithm={algorithm}
            />
          )}
        </div>

        {/* Playback */}
        <div className="mt-auto border-t border-slate-800 pt-3 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-[10px] uppercase tracking-wider font-semibold">
            <span>Playback</span>
            <span>{(progress * 100).toFixed(0)}%</span>
          </div>

          <div
            className="relative h-2 bg-slate-800 rounded-full overflow-hidden cursor-pointer"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              setProgress(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)));
            }}
          >
            <div
              className={clsx(
                "absolute top-0 left-0 h-full",
                (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
                  ? "bg-gradient-to-r from-violet-500 to-purple-400"
                  : "bg-gradient-to-r from-green-500 to-emerald-400"
              )}
              style={{ width: `${progress * 100}%` }}
            />
            {timeline?.steps.map((step, i) => (
              <div
                key={i}
                className="absolute top-0 w-px h-full bg-slate-600/50"
                style={{ left: `${(step.time / timeline.duration) * 100}%` }}
              />
            ))}
          </div>

          <div className="flex items-center justify-center gap-1.5">
            <button onClick={stepBackward} className="p-1.5 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white">
              <ChevronLeft size={18} />
            </button>
            <button onClick={reset} className="p-1.5 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white">
              <RotateCcw size={16} />
            </button>
            <button
              onClick={togglePlay}
              className={clsx(
                "p-2.5 text-slate-900 rounded-full shadow-lg",
                (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
                  ? "bg-violet-500 hover:bg-violet-400"
                  : "bg-green-500 hover:bg-green-400"
              )}
            >
              {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" className="ml-0.5" />}
            </button>
            <button
              onClick={toggleLoop}
              className={clsx(
                "p-1.5 rounded-full",
                isLooping
                  ? (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
                    ? "text-violet-400 bg-violet-900/30"
                    : "text-green-400 bg-green-900/30"
                  : "text-slate-400 hover:bg-slate-800"
              )}
            >
              <Repeat size={16} />
            </button>
            <button onClick={stepForward} className="p-1.5 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white">
              <ChevronRight size={18} />
            </button>
          </div>

          <div className="flex items-center gap-2 justify-center">
            <span className="text-[10px] text-slate-500">Speed</span>
            <input
              type="range" min="0.25" max="2" step="0.25"
              value={speed} onChange={e => setSpeed(parseFloat(e.target.value))}
              className={clsx(
                "w-20 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer",
                (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
                  ? "accent-violet-500"
                  : "accent-green-500"
              )}
            />
            <span className="text-[10px] text-slate-400 w-6">{speed}x</span>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="flex-1 relative bg-slate-950 flex flex-col">
        <div className="absolute top-3 left-3 right-3 flex justify-between items-start pointer-events-none z-10">
          <div>
            <h2 className="text-lg font-light text-slate-200">{operation}</h2>
            <p className={clsx(
              "text-sm",
              (algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot)
                ? "text-violet-400"
                : "text-slate-500"
            )}>{algorithm}</p>
          </div>
          <div className="bg-slate-900/80 backdrop-blur border border-slate-800 p-2 rounded-lg text-right shadow-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Topology</div>
            <div className="text-sm font-semibold text-slate-300">
              {topology === Topology.FourGPU ? "4× H100 (Example)" :
                topology === Topology.SingleNode ? "8× H100 NVLink" : "2×8 H100 + IB"}
            </div>
            {isNVLSAlgo && (
              <div className="text-[9px] text-violet-400 mt-1 flex items-center justify-end gap-1">
                <Zap size={10} /> NVSwitch 3.0 SHARP
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center overflow-hidden">
          <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="max-w-5xl w-full h-auto select-none">
            <defs>
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="glow-strong" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="5" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="glow-purple" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="8" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <radialGradient id="switchGradient" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#6366f1" stopOpacity="0.3" />
              </radialGradient>
            </defs>

            {/* Node backgrounds */}
            {topology === Topology.MultiNode && (
              <>
                <rect x="50" y="130" width="300" height="340" rx="14" className="fill-slate-900/30 stroke-slate-800" strokeWidth="1" />
                <text x="200" y="122" textAnchor="middle" className="fill-slate-600 text-[9px] font-bold uppercase tracking-widest">Node 0 (NVSwitch)</text>

                <rect x="450" y="130" width="300" height="340" rx="14" className="fill-slate-900/30 stroke-slate-800" strokeWidth="1" />
                <text x="600" y="122" textAnchor="middle" className="fill-slate-600 text-[9px] font-bold uppercase tracking-widest">Node 1 (NVSwitch)</text>
              </>
            )}

            {/* NVSwitch visualization (for NVLS/MultiShot) */}
            {isNVLSAlgo && (topology === Topology.SingleNode || topology === Topology.FourGPU) && (
              <g transform={`translate(${switchX}, ${switchY})`}>
                {/* Switch glow when active */}
                {hasSwitchActivity && (
                  <circle r={60} fill="url(#switchGradient)" filter="url(#glow-purple)" className="animate-pulse" />
                )}

                {/* Switch body */}
                <rect x={-40} y={-25} width={80} height={50} rx={8}
                  className={clsx(
                    "fill-slate-800 stroke-2 transition-all duration-300",
                    hasSwitchActivity ? "stroke-violet-400" : "stroke-slate-700"
                  )}
                />

                {/* Switch label */}
                <text y={-32} textAnchor="middle" className="fill-violet-400 text-[9px] font-bold uppercase tracking-wider">
                  NVSwitch
                </text>
                <text y={4} textAnchor="middle" className={clsx(
                  "text-[10px] font-semibold",
                  hasSwitchActivity ? "fill-violet-300" : "fill-slate-500"
                )}>
                  {activeSwitchEvents.some(e => e.type === 'switch-reduce') ? 'REDUCING' :
                    activeSwitchEvents.some(e => e.type === 'switch-route') ? 'ROUTING' :
                      activeSwitchEvents.some(e => e.type.includes('switch') || e.type.includes('multicast')) ? 'ACTIVE' : 'SHARP 3.0'}
                </text>
                <text y={18} textAnchor="middle" className="fill-slate-600 text-[8px]">
                  400 GFlops FP32
                </text>
              </g>
            )}

            {/* Links */}
            {links.map((link, i) => {
              const start = nodes.find(n => n.id === link.from);
              const end = nodes.find(n => n.id === link.to);
              if (!start || !end) return null;

              const cwKey = `${link.from}-${link.to}`;
              const ccwKey = `${link.to}-${link.from}`;
              const isActiveCW = activeLinks.has(cwKey);
              const isActiveCCW = activeLinks.has(ccwKey);

              const isMesh = link.type === 'mesh' || link.type === 'inter-mesh';
              const isInter = link.type === 'inter' || link.type === 'inter-mesh';

              const dx = end.x - start.x;
              const dy = end.y - start.y;
              const len = Math.sqrt(dx * dx + dy * dy);
              const perpX = -dy / len * 3;
              const perpY = dx / len * 3;

              return (
                <g key={i}>
                  <line
                    x1={start.x} y1={start.y}
                    x2={end.x} y2={end.y}
                    className={clsx(
                      "transition-all duration-150",
                      isMesh ? "stroke-slate-800/20" :
                        isInter ? "stroke-slate-700/70" : "stroke-slate-700/80"
                    )}
                    strokeWidth={isMesh ? 1 : (isInter ? 2 : 2.5)}
                    strokeDasharray={isInter ? "3 3" : undefined}
                  />

                  {isActiveCW && (
                    <line
                      x1={start.x + perpX} y1={start.y + perpY}
                      x2={end.x + perpX} y2={end.y + perpY}
                      className="stroke-green-400"
                      strokeWidth={3}
                      strokeLinecap="round"
                      filter="url(#glow)"
                      opacity={0.9}
                    />
                  )}

                  {isActiveCCW && (
                    <line
                      x1={end.x - perpX} y1={end.y - perpY}
                      x2={start.x - perpX} y2={start.y - perpY}
                      className="stroke-blue-400"
                      strokeWidth={3}
                      strokeLinecap="round"
                      filter="url(#glow)"
                      opacity={0.9}
                    />
                  )}
                </g>
              );
            })}

            {/* Switch connections (for NVLS) */}
            {isNVLSAlgo && (topology === Topology.SingleNode || topology === Topology.FourGPU) && activeSwitchEvents.map((evt, i) => {
              if (evt.type === 'to-switch' && evt.from !== undefined) {
                const gpu = nodes.find(n => n.id === evt.from);
                if (!gpu) return null;
                const t = evt.t;
                const x = gpu.x + (switchX - gpu.x) * t;
                const y = gpu.y + (switchY - gpu.y) * t;
                return (
                  <g key={`ts-${i}`}>
                    <line x1={gpu.x} y1={gpu.y} x2={switchX} y2={switchY}
                      className="stroke-violet-400" strokeWidth={2} strokeDasharray="4 2" opacity={0.5} />
                    <circle cx={x} cy={y} r={8} fill={evt.color} filter="url(#glow-strong)" />
                    <text x={x} y={y - 12} textAnchor="middle" className="fill-white text-[8px] font-bold">
                      {evt.label}
                    </text>
                  </g>
                );
              }
              if (evt.type === 'from-switch' && evt.destinations) {
                return evt.destinations.map((dest, j) => {
                  const gpu = nodes.find(n => n.id === dest);
                  if (!gpu) return null;
                  const t = evt.t;
                  const x = switchX + (gpu.x - switchX) * t;
                  const y = switchY + (gpu.y - switchY) * t;
                  return (
                    <g key={`fs-${i}-${j}`}>
                      <line x1={switchX} y1={switchY} x2={gpu.x} y2={gpu.y}
                        className="stroke-green-400" strokeWidth={2} strokeDasharray="4 2" opacity={0.5} />
                      <circle cx={x} cy={y} r={7} fill="#22c55e" filter="url(#glow-strong)" />
                    </g>
                  );
                });
              }
              if (evt.type === 'from-switch-single' && evt.destination !== undefined) {
                const gpu = nodes.find(n => n.id === evt.destination);
                if (!gpu) return null;
                const t = evt.t;
                const x = switchX + (gpu.x - switchX) * t;
                const y = switchY + (gpu.y - switchY) * t;
                return (
                  <g key={`fss-${i}`}>
                    <line x1={switchX} y1={switchY} x2={gpu.x} y2={gpu.y}
                      className="stroke-cyan-400" strokeWidth={2} strokeDasharray="4 2" opacity={0.5} />
                    <circle cx={x} cy={y} r={7} fill={evt.color} filter="url(#glow-strong)" />
                    <text x={x} y={y - 12} textAnchor="middle" className="fill-white text-[8px] font-bold">
                      {evt.label}
                    </text>
                  </g>
                );
              }
              if (evt.type === 'switch-broadcast') {
                // Show broadcast lines to all GPUs
                return nodes.map((gpu, j) => {
                  const t = evt.t;
                  const x = switchX + (gpu.x - switchX) * t;
                  const y = switchY + (gpu.y - switchY) * t;
                  return (
                    <g key={`sb-${i}-${j}`}>
                      <line x1={switchX} y1={switchY} x2={gpu.x} y2={gpu.y}
                        className="stroke-green-400" strokeWidth={1.5} strokeDasharray="4 2" opacity={0.4} />
                      <circle cx={x} cy={y} r={5} fill="#22c55e" opacity={0.8} />
                    </g>
                  );
                });
              }
              return null;
            })}

            {/* Packets */}
            {activePackets.map((pkt) => {
              const start = nodes.find(n => n.id === pkt.from);
              const end = nodes.find(n => n.id === pkt.to);
              if (!start || !end) return null;

              const dx = end.x - start.x;
              const dy = end.y - start.y;
              const len = Math.sqrt(dx * dx + dy * dy);
              const offset = pkt.direction === 'ccw' ? -4 : 4;
              const perpX = -dy / len * offset;
              const perpY = dx / len * offset;

              const x = start.x + (end.x - start.x) * pkt.t + perpX;
              const y = start.y + (end.y - start.y) * pkt.t + perpY;

              return (
                <g key={pkt.id} transform={`translate(${x}, ${y})`}>
                  <circle
                    r={topology === Topology.MultiNode ? 7 : 9}
                    fill={pkt.color}
                    filter="url(#glow-strong)"
                    stroke={pkt.direction === 'ccw' ? '#3b82f6' : '#22c55e'}
                    strokeWidth={1.5}
                  />
                  <text
                    y={-12}
                    textAnchor="middle"
                    className="fill-white text-[8px] font-bold"
                    style={{ textShadow: '0 1px 3px rgba(0,0,0,0.9)' }}
                  >
                    {pkt.label}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map(node => {
              const isComputing = activeComputes.some(c => c.nodeId === node.id);
              const computeOp = activeComputes.find(c => c.nodeId === node.id);
              const size = topology === Topology.MultiNode ? 28 : 36;
              const offset = size / 2;

              // Check if this node is involved in switch activity
              const inSwitchActivity = activeSwitchEvents.some(e =>
                e.from === node.id ||
                (e.destinations && e.destinations.includes(node.id)) ||
                (e.destination === node.id) ||
                (e.sources && e.sources.includes(node.id))
              );

              return (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                  {isComputing && (
                    <circle r={size * 0.7} className="fill-green-500/20 animate-pulse" />
                  )}
                  {inSwitchActivity && (
                    <circle r={size * 0.65} className="fill-violet-500/20 animate-pulse" />
                  )}

                  <rect
                    x={-offset} y={-offset} width={size} height={size} rx={4}
                    className={clsx(
                      "fill-slate-900 transition-all duration-150",
                      isComputing ? "stroke-green-400" :
                        inSwitchActivity ? "stroke-violet-400" : "stroke-slate-700"
                    )}
                    strokeWidth={2}
                  />

                  <rect
                    x={-offset * 0.55} y={-offset * 0.55}
                    width={size * 0.55} height={size * 0.55}
                    rx={2}
                    className="fill-slate-800"
                  />

                  <text
                    y={offset + 12}
                    textAnchor="middle"
                    className="fill-slate-500 text-[8px] font-semibold"
                  >
                    {node.label}
                  </text>

                  {isComputing && computeOp && (
                    <text y={4} textAnchor="middle" className="fill-green-400 text-[10px] font-bold">
                      {computeOp.label}
                    </text>
                  )}
                </g>
              );
            })}



            {(algorithm === Algorithm.NVLS || algorithm === Algorithm.MultiShot) && (topology === Topology.SingleNode || topology === Topology.FourGPU) && (
              <g transform="translate(400, 520)">
                <text x={0} y={0} textAnchor="middle" className="fill-violet-400 text-[10px] font-medium">
                  {algorithm === Algorithm.NVLS
                    ? '⚡ NVSwitch SHARP: In-Network Reduction + Multicast'
                    : '⚡ MultiShot: ReduceScatter (NVLS) + AllGather (NVLS)'}
                </text>
              </g>
            )}
          </svg>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 right-3 flex justify-center pointer-events-none">
          <div className={clsx(
            "flex gap-4 backdrop-blur px-4 py-2 rounded-full border text-[10px] shadow-xl",
            isNVLSAlgo
              ? "bg-violet-900/60 border-violet-700 text-violet-200"
              : "bg-slate-900/80 border-slate-800 text-slate-400"
          )}>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-slate-900 border border-slate-700"></div>
              <span>Idle</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-slate-900 border-2 border-green-400"></div>
              <span>SM Compute</span>
            </div>
            {isNVLSAlgo && (
              <>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded bg-violet-500/50 border border-violet-400"></div>
                  <span>Switch I/O</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Zap size={12} className="text-violet-300" />
                  <span>In-Switch Reduce</span>
                </div>
              </>
            )}
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]"></div>
              <span>Data Chunk</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
