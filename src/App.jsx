import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Play, Pause, RotateCcw, Repeat, ChevronRight, ChevronLeft, Eye, EyeOff, BarChart3 } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

// === TYPES & CONSTANTS ===
const Operation = {
  AllReduce: 'AllReduce',
  Reduce: 'Reduce',
  Broadcast: 'Broadcast',
  AllGather: 'AllGather',
  ReduceScatter: 'ReduceScatter'
};

const Algorithm = {
  Naive: 'Naive (Parameter Server)',
  Ring: 'Ring (Single Direction)',
  BiRing: 'Ring (Bidirectional)'
};

const Topology = {
  SingleNode: 'Single Node (8x H100)',
  MultiNode: 'Multi Node (2x8 H100)'
};

// Distinct colors for chunks
const CHUNK_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e',
  '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899',
  '#f43f5e', '#84cc16', '#14b8a6', '#6366f1',
  '#a855f7', '#f59e0b', '#10b981', '#0ea5e9'
];

// === SIMULATION ENGINE ===
const generateTimeline = (op, algo, topo) => {
  const NODE_COUNT = topo === Topology.MultiNode ? 16 : 8;
  const CHUNK_COUNT = NODE_COUNT;

  const events = [];
  const steps = [];
  const bandwidthSamples = []; // { time, linkId, utilization }
  let currentTime = 0;

  const TRANSFER_TIME = 400;
  const INTER_NODE_MULTIPLIER = 1.5;
  const COMPUTE_TIME = 120;

  // Track buffer state over time
  // bufferStates[time] = { gpuId: [{ chunkId, state, partial? }] }
  const bufferSnapshots = [];

  // Initialize buffer state
  const initBufferState = () => {
    const state = {};
    for (let i = 0; i < NODE_COUNT; i++) {
      // Each GPU starts with its own local data (conceptually all chunks, but only owns chunk i)
      state[i] = Array.from({ length: CHUNK_COUNT }, (_, c) => ({
        chunkId: c,
        state: 'local', // local | partial | reduced | received | final
        reductionCount: 1 // How many GPUs' data is included
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

  // Ring helpers
  const ringNext = (i) => {
    if (topo === Topology.MultiNode) {
      if (i === 7) return 8;
      if (i === 15) return 0;
      return i + 1;
    }
    return (i + 1) % NODE_COUNT;
  };

  const ringPrev = (i) => {
    if (topo === Topology.MultiNode) {
      if (i === 8) return 7;
      if (i === 0) return 15;
      return i - 1;
    }
    return (i - 1 + NODE_COUNT) % NODE_COUNT;
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

    // Record bandwidth utilization
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

  // === ALGORITHM IMPLEMENTATIONS ===

  if (algo === Algorithm.Naive) {
    const ROOT = 0;

    if (op === Operation.AllReduce) {
      addStep('Initial: Each GPU has local gradient data', 'init');
      currentTime += 300;

      // Gather phase - all send to root (BOTTLENECK!)
      addStep('Gather: All GPUs send to root. Root receives sequentially → bandwidth bottleneck!', 'gather');

      // Show all transfers starting simultaneously
      let maxDuration = 0;
      for (let i = 1; i < NODE_COUNT; i++) {
        const d = addTransfer(i, ROOT, i, `G${i}`);
        addLinkActive(i, ROOT, d);
        maxDuration = Math.max(maxDuration, d);
      }

      // But root can only receive one at a time effectively
      // Record bandwidth showing root link saturated
      for (let i = 1; i < NODE_COUNT; i++) {
        bandwidthSamples.push({
          time: currentTime + (i - 1) * TRANSFER_TIME * 0.3,
          endTime: currentTime + i * TRANSFER_TIME * 0.3,
          linkId: `0-${i}`,
          from: i,
          to: 0,
          utilization: 1.0,
          isBottleneck: true
        });
      }

      currentTime += TRANSFER_TIME * (NODE_COUNT - 1) * 0.35;

      // Update buffer - root now has all data
      bufferState[ROOT] = [{ chunkId: 'all', state: 'gathered', reductionCount: NODE_COUNT }];

      addStep('Reduce: Root computes sum of all gradients', 'reduce');
      addCompute(ROOT, 'Σ', COMPUTE_TIME * 2);
      currentTime += COMPUTE_TIME * 2;

      bufferState[ROOT] = [{ chunkId: 'all', state: 'reduced', reductionCount: NODE_COUNT }];

      addStep('Broadcast: Root sends result to all. Another bottleneck at root!', 'broadcast');

      for (let i = 1; i < NODE_COUNT; i++) {
        const d = addTransfer(ROOT, i, 0, 'Σ');
        addLinkActive(ROOT, i, d);
      }
      currentTime += TRANSFER_TIME * (NODE_COUNT - 1) * 0.35;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 'all', state: 'final', reductionCount: NODE_COUNT }];
      }
      addStep('Complete. Total time: O(N) due to root bottleneck. Bandwidth wasted!', 'done');
    }
    else if (op === Operation.Reduce) {
      addStep('Initial: Each GPU has local data to reduce', 'init');
      currentTime += 300;

      // Gather phase - all send to root (BOTTLENECK!)
      addStep('Gather: All GPUs send to root (GPU 0). Sequential bottleneck!', 'gather');

      for (let i = 1; i < NODE_COUNT; i++) {
        const d = addTransfer(i, ROOT, i, `D${i}`);
        addLinkActive(i, ROOT, d);
      }

      currentTime += TRANSFER_TIME * (NODE_COUNT - 1) * 0.35;

      bufferState[ROOT] = [{ chunkId: 'all', state: 'gathered', reductionCount: NODE_COUNT }];

      addStep('Reduce: Root computes sum of all data', 'reduce');
      addCompute(ROOT, 'Σ', COMPUTE_TIME * 2);
      currentTime += COMPUTE_TIME * 2;

      bufferState[ROOT] = [{ chunkId: 'all', state: 'reduced', reductionCount: NODE_COUNT }];

      // Other GPUs don't have the result
      for (let i = 1; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: i, state: 'sent', reductionCount: 1 }];
      }

      addStep('Complete. Only root (GPU 0) has the reduced result. O(N) time.', 'done');
    }
    else if (op === Operation.AllGather) {
      addStep('Initial: Each GPU has unique data chunk i', 'init');
      currentTime += 300;

      addStep('Naive: All-to-all direct transfers. Network congestion!', 'transfer');
      for (let src = 0; src < NODE_COUNT; src++) {
        for (let dst = 0; dst < NODE_COUNT; dst++) {
          if (src !== dst) {
            addTransfer(src, dst, src, `D${src}`);
          }
        }
      }
      currentTime += TRANSFER_TIME * 2;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: 1
        }));
      }
      addStep('Complete. Network was congested - not bandwidth optimal.', 'done');
    }
    else if (op === Operation.ReduceScatter) {
      addStep('Initial: Each GPU has full vector', 'init');
      currentTime += 300;

      addStep('Naive: Everyone sends portions to chunk owners', 'transfer');
      for (let owner = 0; owner < NODE_COUNT; owner++) {
        for (let src = 0; src < NODE_COUNT; src++) {
          if (src !== owner) {
            addTransfer(src, owner, owner, `→${owner}`);
          }
        }
      }
      currentTime += TRANSFER_TIME * 2;

      addStep('Each GPU reduces received data', 'reduce');
      for (let i = 0; i < NODE_COUNT; i++) {
        addCompute(i, 'Σ');
      }
      currentTime += COMPUTE_TIME;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: i, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      addStep('Complete. Congestion at each destination GPU.', 'done');
    }
    else if (op === Operation.Broadcast) {
      addStep('Initial: Root has data to broadcast', 'init');
      bufferState[0] = [{ chunkId: 0, state: 'source', reductionCount: 1 }];
      currentTime += 300;

      addStep('Naive: Root sends to everyone directly (bottleneck!)', 'transfer');
      for (let i = 1; i < NODE_COUNT; i++) {
        addTransfer(0, i, 0, 'Data');
        addLinkActive(0, i, TRANSFER_TIME);
      }
      currentTime += TRANSFER_TIME * (NODE_COUNT - 1) * 0.35;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 0, state: 'final', reductionCount: 1 }];
      }
      addStep('Complete. Root bandwidth was the bottleneck.', 'done');
    }
  }
  else if (algo === Algorithm.Ring || algo === Algorithm.BiRing) {
    const isBidirectional = algo === Algorithm.BiRing;
    const ROOT = 0;

    if (op === Operation.AllReduce) {
      addStep('Initial: Each GPU has local gradient. Data split into N chunks logically.', 'init');
      saveBufferSnapshot();
      currentTime += 300;

      if (isBidirectional) {
        addStep('Using TWO rings: clockwise reduces chunks 0-3, counter-clockwise reduces chunks 4-7. 2× bandwidth!', 'init');
        currentTime += 200;
      }

      // === REDUCE-SCATTER PHASE ===
      const stepsNeeded = NODE_COUNT - 1;

      for (let step = 0; step < stepsNeeded; step++) {
        if (isBidirectional) {
          addStep(
            `Reduce-Scatter ${step + 1}/${stepsNeeded}: CW ring sends even chunks, CCW ring sends odd chunks simultaneously`,
            'reduce-scatter'
          );
        } else {
          addStep(
            `Reduce-Scatter ${step + 1}/${stepsNeeded}: Each GPU sends chunk to next neighbor, reduces incoming`,
            'reduce-scatter'
          );
        }

        let maxDuration = 0;

        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          // Clockwise ring
          const sendChunkCW = ((gpu - step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
          const sendToCW = ringNext(gpu);

          if (isBidirectional) {
            // Only send half the chunks on each ring
            if (sendChunkCW < NODE_COUNT / 2) {
              const d = addTransfer(gpu, sendToCW, sendChunkCW, `C${sendChunkCW}`, 'cw');
              addLinkActive(gpu, sendToCW, d, 'cw');
              maxDuration = Math.max(maxDuration, d);
            }

            // Counter-clockwise ring for other half
            const sendChunkCCW = ((gpu + step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
            const sendToCCW = ringPrev(gpu);

            if (sendChunkCCW >= NODE_COUNT / 2) {
              const d = addTransfer(gpu, sendToCCW, sendChunkCCW, `C${sendChunkCCW}`, 'ccw');
              addLinkActive(gpu, sendToCCW, d, 'ccw');
              maxDuration = Math.max(maxDuration, d);
            }
          } else {
            const d = addTransfer(gpu, sendToCW, sendChunkCW, `C${sendChunkCW}`, 'cw');
            addLinkActive(gpu, sendToCW, d, 'cw');
            maxDuration = Math.max(maxDuration, d);
          }
        }

        currentTime += maxDuration;

        // Compute reduction
        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          addCompute(gpu, '+', COMPUTE_TIME * 0.5);
        }
        currentTime += COMPUTE_TIME * 0.5;

        // Update buffer state - show partial reductions
        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          const ownedChunk = (gpu + 1) % NODE_COUNT;
          const reductionsCompleted = step + 2; // Started with 1, now have step+2
          bufferState[gpu] = bufferState[gpu].map((chunk, idx) => {
            if (idx === ownedChunk) {
              return {
                ...chunk,
                state: reductionsCompleted >= NODE_COUNT ? 'reduced' : 'partial',
                reductionCount: Math.min(reductionsCompleted, NODE_COUNT)
              };
            }
            return chunk;
          });
        }
        saveBufferSnapshot();
      }

      addStep('Reduce-Scatter done! Each GPU has fully reduced chunk i', 'reduce-scatter-done');

      // Simplify buffer view
      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        bufferState[gpu] = [{ chunkId: (gpu + 1) % NODE_COUNT, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      saveBufferSnapshot();
      currentTime += 200;

      // === ALL-GATHER PHASE ===
      for (let step = 0; step < stepsNeeded; step++) {
        if (isBidirectional) {
          addStep(
            `AllGather ${step + 1}/${stepsNeeded}: Both rings forward reduced chunks simultaneously`,
            'all-gather'
          );
        } else {
          addStep(
            `AllGather ${step + 1}/${stepsNeeded}: Forward reduced chunks around the ring`,
            'all-gather'
          );
        }

        let maxDuration = 0;

        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          const sendChunkCW = ((gpu - step + 1) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
          const sendToCW = ringNext(gpu);

          if (isBidirectional) {
            if (sendChunkCW < NODE_COUNT / 2) {
              const d = addTransfer(gpu, sendToCW, sendChunkCW, `C${sendChunkCW}`, 'cw');
              addLinkActive(gpu, sendToCW, d, 'cw');
              maxDuration = Math.max(maxDuration, d);
            }

            const sendChunkCCW = ((gpu + step - 1) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
            const sendToCCW = ringPrev(gpu);

            if (sendChunkCCW >= NODE_COUNT / 2) {
              const d = addTransfer(gpu, sendToCCW, sendChunkCCW, `C${sendChunkCCW}`, 'ccw');
              addLinkActive(gpu, sendToCCW, d, 'ccw');
              maxDuration = Math.max(maxDuration, d);
            }
          } else {
            const d = addTransfer(gpu, sendToCW, sendChunkCW, `C${sendChunkCW}`, 'cw');
            addLinkActive(gpu, sendToCW, d, 'cw');
            maxDuration = Math.max(maxDuration, d);
          }
        }

        currentTime += maxDuration;

        // Update buffer - GPUs accumulate final chunks
        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          const chunksReceived = step + 2;
          bufferState[gpu] = Array.from({ length: Math.min(chunksReceived, NODE_COUNT) }, (_, i) => ({
            chunkId: (gpu - i + NODE_COUNT) % NODE_COUNT,
            state: 'final',
            reductionCount: NODE_COUNT
          }));
        }
        saveBufferSnapshot();
      }

      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        bufferState[gpu] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: NODE_COUNT
        }));
      }

      const bwNote = isBidirectional
        ? 'Bidirectional achieves 2× bandwidth utilization!'
        : 'Data moved: 2×(N-1)/N × Size (bandwidth optimal)';
      addStep(`AllReduce Complete! ${bwNote}`, 'done');
    }
    else if (op === Operation.Reduce) {
      addStep('Initial: Each GPU has local data. Goal: Reduce to root (GPU 0) only.', 'init');
      saveBufferSnapshot();
      currentTime += 300;

      if (isBidirectional) {
        addStep('Ring Reduce: Pipeline reduction around ring toward root. Using both directions.', 'init');
        currentTime += 200;

        // Bidirectional: two halves of GPUs converge on root
        const halfSteps = Math.ceil(NODE_COUNT / 2);

        for (let step = 0; step < halfSteps; step++) {
          addStep(`Reduce step ${step + 1}/${halfSteps}: Both directions converge toward root`, 'reduce');

          let maxDuration = 0;

          // CW direction: GPUs 1, 2, ..., N/2 send toward 0
          // Logic: Pipeline data towards 0.
          // Step 0: 1->0
          // Step 1: 2->1, 1->0
          // ...
          for (let gpu = 1; gpu <= NODE_COUNT / 2; gpu++) {
            const distance = gpu - 1; // 1 is dist 0 (immediate), 2 is dist 1...
            if (distance <= step) {
              const sender = gpu;
              const receiver = sender - 1;
              const d = addTransfer(sender, receiver, 0, `Σ`, 'cw');
              addLinkActive(sender, receiver, d, 'cw');
              maxDuration = Math.max(maxDuration, d);
            }
          }

          // CCW direction: GPUs N-1, N-2, ..., N/2+1 send toward 0
          // Logic: Pipeline data towards 0 via N-1.
          // Step 0: 7->0
          // Step 1: 6->7, 7->0
          // ...
          for (let gpu = NODE_COUNT - 1; gpu > NODE_COUNT / 2; gpu--) {
            const distance = NODE_COUNT - 1 - gpu; // 7 is dist 0, 6 is dist 1...
            if (distance <= step) {
              const sender = gpu;
              const receiver = ringNext(sender);
              const d = addTransfer(sender, receiver, 0, `Σ`, 'ccw');
              addLinkActive(sender, receiver, d, 'ccw');
              maxDuration = Math.max(maxDuration, d);
            }
          }

          if (maxDuration > 0) {
            currentTime += maxDuration;

            // Compute reduction at receiving GPUs
            for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
              addCompute(gpu, '+', COMPUTE_TIME * 0.5);
            }
            currentTime += COMPUTE_TIME * 0.5;
          }
          saveBufferSnapshot();
        }
      } else {
        // Single direction ring reduce: pipeline around the ring
        addStep('Ring Reduce: Each GPU receives from prev, reduces, forwards to next', 'reduce');

        const stepsNeeded = NODE_COUNT - 1;

        for (let step = 0; step < stepsNeeded; step++) {
          // GPU (N-1-step) sends to GPU (N-2-step), etc., pipelining toward GPU 0
          const sender = NODE_COUNT - 1 - step;
          const receiver = ringPrev(sender);

          addStep(`Step ${step + 1}/${stepsNeeded}: GPU ${sender} → GPU ${receiver} (accumulating partial sums)`, 'reduce');

          const d = addTransfer(sender, receiver, 0, step === 0 ? `D${sender}` : 'Σ', 'cw');
          addLinkActive(sender, receiver, d, 'cw');

          currentTime += d * 0.4; // Pipelined

          // Reduction at receiver
          addCompute(receiver, '+', COMPUTE_TIME * 0.3);

          // Update buffer state
          bufferState[receiver] = [{
            chunkId: 'partial',
            state: 'partial',
            reductionCount: step + 2
          }];
          bufferState[sender] = [{ chunkId: sender, state: 'sent', reductionCount: 1 }];

          saveBufferSnapshot();
        }

        currentTime += TRANSFER_TIME * 0.6 + COMPUTE_TIME * 0.5;
      }

      // Final state
      bufferState[ROOT] = [{ chunkId: 'all', state: 'reduced', reductionCount: NODE_COUNT }];
      for (let i = 1; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: i, state: 'sent', reductionCount: 1 }];
      }

      const bwNote = isBidirectional
        ? 'Bidirectional reduces latency by half!'
        : 'Latency: O(N), but pipelined for large messages';
      addStep(`Reduce Complete! Only root has result. ${bwNote}`, 'done');
    }
    else if (op === Operation.AllGather) {
      addStep('Initial: Each GPU i has unique chunk i', 'init');
      currentTime += 300;

      const stepsNeeded = NODE_COUNT - 1;

      for (let step = 0; step < stepsNeeded; step++) {
        addStep(`Ring AllGather ${step + 1}/${stepsNeeded}: Pass chunks around the ring`, 'all-gather');

        let maxDuration = 0;
        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          const sendChunk = ((gpu - step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
          const sendTo = ringNext(gpu);

          const d = addTransfer(gpu, sendTo, sendChunk, `C${sendChunk}`, 'cw');
          addLinkActive(gpu, sendTo, d, 'cw');
          maxDuration = Math.max(maxDuration, d);

          if (isBidirectional) {
            const sendChunkCCW = ((gpu + step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
            const sendToCCW = ringPrev(gpu);
            const d2 = addTransfer(gpu, sendToCCW, sendChunkCCW, `C${sendChunkCCW}`, 'ccw');
            addLinkActive(gpu, sendToCCW, d2, 'ccw');
          }
        }
        currentTime += maxDuration;
        saveBufferSnapshot();
      }

      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        bufferState[gpu] = Array.from({ length: NODE_COUNT }, (_, c) => ({
          chunkId: c, state: 'final', reductionCount: 1
        }));
      }
      addStep('AllGather Complete! Data: (N-1)/N × Size', 'done');
    }
    else if (op === Operation.ReduceScatter) {
      addStep('Initial: Each GPU has full data vector', 'init');
      currentTime += 300;

      const stepsNeeded = NODE_COUNT - 1;

      for (let step = 0; step < stepsNeeded; step++) {
        addStep(`Ring ReduceScatter ${step + 1}/${stepsNeeded}: Send and reduce`, 'reduce-scatter');

        let maxDuration = 0;
        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          const sendChunk = ((gpu - step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
          const sendTo = ringNext(gpu);

          const d = addTransfer(gpu, sendTo, sendChunk, `C${sendChunk}`, 'cw');
          addLinkActive(gpu, sendTo, d, 'cw');
          maxDuration = Math.max(maxDuration, d);

          if (isBidirectional) {
            const sendChunkCCW = ((gpu + step) % NODE_COUNT + NODE_COUNT) % NODE_COUNT;
            const sendToCCW = ringPrev(gpu);
            const d2 = addTransfer(gpu, sendToCCW, sendChunkCCW, `C${sendChunkCCW}`, 'ccw');
            addLinkActive(gpu, sendToCCW, d2, 'ccw');
          }
        }
        currentTime += maxDuration;

        for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
          addCompute(gpu, '+', COMPUTE_TIME * 0.5);
        }
        currentTime += COMPUTE_TIME * 0.5;
        saveBufferSnapshot();
      }

      for (let gpu = 0; gpu < NODE_COUNT; gpu++) {
        bufferState[gpu] = [{ chunkId: (gpu + 1) % NODE_COUNT, state: 'reduced', reductionCount: NODE_COUNT }];
      }
      addStep('ReduceScatter Complete! Each GPU owns reduced chunk i', 'done');
    }
    else if (op === Operation.Broadcast) {
      addStep('Initial: Root (GPU 0) has data', 'init');
      bufferState[0] = [{ chunkId: 0, state: 'source', reductionCount: 1 }];
      currentTime += 300;

      addStep('Ring Broadcast: Pipeline through ring (lower latency than tree for small data)', 'transfer');

      for (let step = 0; step < NODE_COUNT - 1; step++) {
        const sender = step;
        const receiver = ringNext(sender);

        const d = addTransfer(sender, receiver, 0, 'Data', 'cw');
        addLinkActive(sender, receiver, d, 'cw');
        currentTime += d * 0.4; // Pipelined

        bufferState[receiver] = [{ chunkId: 0, state: 'received', reductionCount: 1 }];
        saveBufferSnapshot();
      }

      currentTime += TRANSFER_TIME * 0.6;

      for (let i = 0; i < NODE_COUNT; i++) {
        bufferState[i] = [{ chunkId: 0, state: 'final', reductionCount: 1 }];
      }
      addStep('Broadcast Complete! Latency: O(N), but pipelined', 'done');
    }
  }

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

  if (topo === Topology.SingleNode) {
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

  // Calculate bandwidth utilization over time
  const bucketCount = 50;
  const bucketDuration = timeline.duration / bucketCount;

  const utilizationData = useMemo(() => {
    const buckets = Array(bucketCount).fill(0);
    const maxLinks = topology === Topology.MultiNode ? 16 : 8; // Ring links

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

        {/* Grid */}
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

          {/* Area chart */}
          <path
            d={`M 0 ${chartHeight} ${utilizationData.map((d, i) =>
              `L ${(i / bucketCount) * chartWidth} ${chartHeight * (1 - d.utilization / maxUtil)}`
            ).join(' ')} L ${chartWidth} ${chartHeight} Z`}
            fill="url(#utilGradient)"
          />

          {/* Line */}
          <path
            d={`M ${utilizationData.map((d, i) =>
              `${(i / bucketCount) * chartWidth} ${chartHeight * (1 - d.utilization / maxUtil)}`
            ).join(' L ')}`}
            fill="none"
            className="stroke-green-400"
            strokeWidth={1.5}
          />

          {/* Current time indicator */}
          <line
            x1={(currentTime / timeline.duration) * chartWidth}
            y1={0}
            x2={(currentTime / timeline.duration) * chartWidth}
            y2={chartHeight}
            className="stroke-white"
            strokeWidth={1}
          />
        </g>

        {/* X-axis labels */}
        <text
          x={padding.left}
          y={height - 5}
          className="fill-slate-500 text-[8px]"
        >
          0
        </text>
        <text
          x={width - padding.right}
          y={height - 5}
          className="fill-slate-500 text-[8px]"
          textAnchor="end"
        >
          {(timeline.duration / 1000).toFixed(1)}s
        </text>
      </svg>
    </div>
  );
};

// === BUFFER STATE COMPONENT ===
const BufferStateView = ({ timeline, currentTime, nodeCount }) => {
  if (!timeline || !timeline.bufferSnapshots.length) return null;

  // Find current buffer state
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
                        isSent ? "opacity-30" : ""
                      )}
                      style={{ backgroundColor: color }}
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
          <div className="w-2 h-2 rounded-sm bg-green-500 ring-1 ring-white/50"></div>
          <span>Final</span>
        </div>
      </div>
    </div>
  );
};

// === MAIN COMPONENT ===
const App = () => {
  const [operation, setOperation] = useState(Operation.AllReduce);
  const [algorithm, setAlgorithm] = useState(Algorithm.Ring);
  const [topology, setTopology] = useState(Topology.SingleNode);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [timeline, setTimeline] = useState(null);
  const [showBuffers, setShowBuffers] = useState(true);
  const [showBandwidth, setShowBandwidth] = useState(true);

  const lastTimeRef = useRef(0);

  useEffect(() => {
    const tl = generateTimeline(operation, algorithm, topology);
    setTimeline(tl);
    setProgress(0);
    setIsPlaying(false);
  }, [operation, algorithm, topology]);

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

  const activePackets = useMemo(() => {
    if (!timeline) return [];
    return timeline.events
      .filter(e => e.type === 'transfer')
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

  return (
    <div className="flex h-screen bg-slate-950 text-white font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 flex-shrink-0 border-r border-slate-800 bg-slate-900/50 p-4 flex flex-col gap-3 overflow-y-auto">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-7 h-7 rounded bg-gradient-to-tr from-green-500 to-emerald-400 flex items-center justify-center font-bold text-slate-900 text-sm">N</div>
          <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-400">NCCL Visualizer</h1>
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
          <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Algorithm</h3>
          <div className="flex flex-col gap-1.5">
            {Object.values(Algorithm).map(algo => (
              <button
                key={algo}
                onClick={() => setAlgorithm(algo)}
                className={clsx(
                  "p-2 text-xs rounded border text-left transition-all flex items-center gap-2",
                  algorithm === algo
                    ? "bg-blue-500/20 border-blue-500 text-blue-300"
                    : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                )}
              >
                <div className={clsx("w-1.5 h-1.5 rounded-full", algorithm === algo ? "bg-blue-400" : "bg-slate-600")} />
                {algo}
              </button>
            ))}
          </div>
        </div>

        {/* Current Step */}
        {currentStep && (
          <div className="bg-slate-800/50 rounded-lg p-2.5 border border-slate-700">
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
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-500 to-emerald-400"
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
              className="p-2.5 bg-green-500 hover:bg-green-400 text-slate-900 rounded-full shadow-lg"
            >
              {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" className="ml-0.5" />}
            </button>
            <button
              onClick={toggleLoop}
              className={clsx("p-1.5 rounded-full", isLooping ? "text-green-400 bg-green-900/30" : "text-slate-400 hover:bg-slate-800")}
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
              className="w-20 accent-green-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
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
            <p className="text-slate-500 text-sm">{algorithm}</p>
          </div>
          <div className="bg-slate-900/80 backdrop-blur border border-slate-800 p-2 rounded-lg text-right shadow-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Topology</div>
            <div className="text-sm font-semibold text-slate-300">
              {topology === Topology.SingleNode ? "8× H100 NVLink" : "2×8 H100 + IB"}
            </div>
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
              <marker id="arrow-cw" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="#22c55e" />
              </marker>
              <marker id="arrow-ccw" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="#3b82f6" />
              </marker>
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

            {/* Links */}
            {links.map((link, i) => {
              const start = nodes.find(n => n.id === link.from);
              const end = nodes.find(n => n.id === link.to);
              if (!start || !end) return null;

              const cwKey = `${link.from}-${link.to}`;
              const ccwKey = `${link.to}-${link.from}`;
              const isActiveCW = activeLinks.has(cwKey);
              const isActiveCCW = activeLinks.has(ccwKey);
              const isActive = isActiveCW || isActiveCCW;

              const isMesh = link.type === 'mesh' || link.type === 'inter-mesh';
              const isInter = link.type === 'inter' || link.type === 'inter-mesh';

              // Offset for bidirectional visualization
              const dx = end.x - start.x;
              const dy = end.y - start.y;
              const len = Math.sqrt(dx * dx + dy * dy);
              const perpX = -dy / len * 3;
              const perpY = dx / len * 3;

              return (
                <g key={i}>
                  {/* Base link */}
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

                  {/* Active CW highlight */}
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

                  {/* Active CCW highlight */}
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

            {/* Packets */}
            {activePackets.map((pkt) => {
              const start = nodes.find(n => n.id === pkt.from);
              const end = nodes.find(n => n.id === pkt.to);
              if (!start || !end) return null;

              // Offset based on direction
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

              return (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                  {isComputing && (
                    <circle r={size * 0.7} className="fill-green-500/20 animate-pulse" />
                  )}

                  <rect
                    x={-offset} y={-offset} width={size} height={size} rx={4}
                    className={clsx(
                      "fill-slate-900 transition-all duration-150",
                      isComputing ? "stroke-green-400" : "stroke-slate-700"
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

            {/* Ring direction indicators */}
            {algorithm === Algorithm.BiRing && (
              <g transform="translate(400, 520)">
                <text x={-60} y={0} className="fill-green-400 text-[10px] font-medium">→ CW Ring</text>
                <text x={20} y={0} className="fill-blue-400 text-[10px] font-medium">← CCW Ring</text>
              </g>
            )}
          </svg>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 right-3 flex justify-center pointer-events-none">
          <div className="flex gap-4 bg-slate-900/80 backdrop-blur px-4 py-2 rounded-full border border-slate-800 text-[10px] text-slate-400 shadow-xl">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-slate-900 border border-slate-700"></div>
              <span>Idle</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-slate-900 border-2 border-green-400"></div>
              <span>Compute</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-green-400"></div>
              <span>CW Transfer</span>
            </div>
            {algorithm === Algorithm.BiRing && (
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-0.5 bg-blue-400"></div>
                <span>CCW Transfer</span>
              </div>
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
