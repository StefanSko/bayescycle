// Pure design-document helpers for the simulation workflow.

const DATA_FORMAT = "bayescycle.data.json.v1";

export function designDocument(design) {
  const variables = Object.fromEntries(
    Object.entries(design).map(([name, row]) => {
      if (Object.hasOwn(row, "value")) {
        if (row.countLike !== false) {
          if (!Number.isInteger(row.value) || row.value < 1) {
            throw new Error(`Design scalar ${name} must be a positive integer`);
          }
          return [name, { dtype: "int64", shape: [], values: [row.value] }];
        }
        if (!Number.isFinite(row.value)) {
          throw new Error(`Design scalar ${name} must be finite`);
        }
        return [name, { dtype: "float64", shape: [], values: [row.value] }];
      }
      const n = row.n ?? 50;
      if (
        !Number.isFinite(row.low) ||
        !Number.isFinite(row.high) ||
        !Number.isInteger(n) ||
        n < 1
      ) {
        throw new Error(
          `Design values for ${name} require finite low/high and a positive integer n`,
        );
      }
      const values = Array.from({ length: n }, (_, index) =>
        n === 1 ? row.low : row.low + ((row.high - row.low) * index) / (n - 1),
      );
      // Name-seeded permutations prevent shared linspaces from making predictors collinear.
      shuffle(values, mulberry32(fnv1a(name)));
      return [name, { dtype: "float64", shape: [n], values }];
    }),
  );
  return { format: DATA_FORMAT, variables };
}

export function truthDocument(truth, sizes = {}) {
  const variables = Object.fromEntries(
    Object.entries(truth).map(([name, value]) => {
      if (!Number.isFinite(value)) {
        throw new Error(`Truth value for ${name} must be finite`);
      }
      const sizeSpec = sizes[name];
      if (sizeSpec === undefined) {
        return [name, { dtype: "float64", shape: [], values: [value] }];
      }
      const size = typeof sizeSpec === "number" ? sizeSpec : sizeSpec.size;
      if (!Number.isInteger(size) || size < 1) {
        throw new Error(`Truth size for ${name} must be a positive integer`);
      }
      const values = typeof sizeSpec === "object" && sizeSpec.ordered === true
        ? Array.from(
            { length: size },
            (_, index) => value + index - (size - 1) / 2,
          )
        : Array(size).fill(value);
      return [name, { dtype: "float64", shape: [size], values }];
    }),
  );
  return { format: DATA_FORMAT, variables };
}

export function defaultTruth(ir) {
  return Object.fromEntries(
    ir.model.params.map((parameter) => [
      parameter.name,
      parameter.value.constraint?.node === "Positive" ? 1 : 0,
    ]),
  );
}

export function designDefaults(ir) {
  const vectors = ir.model.data.filter((input) => {
    const schema = input.value.schema;
    return schema.dims?.length === 1 || schema.rank === 1;
  });
  const derivedScalars = new Set(
    vectors.flatMap((input) => input.value.schema.dims?.map((dim) => dim.name) ?? []),
  );
  const countLikeScalars = new Set(
    ir.model.data.flatMap(
      (input) => input.value.schema.dims?.map((dim) => dim.name) ?? [],
    ),
  );
  for (const parameter of ir.model.params) {
    const size = parameter.value.size;
    if (size?.node === "DataRef") countLikeScalars.add(size.name);
  }
  return Object.fromEntries(
    ir.model.data
      .filter((input) => {
        if (vectors.includes(input)) return true;
        const schema = input.value.schema;
        const scalar = (schema.dims?.length ?? 0) === 0 && !(schema.rank > 0);
        return scalar && !derivedScalars.has(input.name);
      })
      .map((input) => [
        input.name,
        vectors.includes(input)
          ? { low: 0.5, high: 1.5, n: 50 }
          : countLikeScalars.has(input.name)
            ? { value: 2, countLike: true }
            : { value: 1, countLike: false },
      ]),
  );
}

function fnv1a(value) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
}
