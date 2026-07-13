export function subsampleReplicates(replicates, cap = 50) {
  if (!Number.isInteger(cap) || cap < 1) {
    throw new Error("Replicate cap must be a positive integer");
  }
  if (replicates.length <= cap) return replicates;
  const step = Math.ceil(replicates.length / cap);
  const sampled = [];
  for (let index = 0; index < replicates.length && sampled.length < cap; index += step) {
    sampled.push(replicates[index]);
  }
  return sampled;
}
