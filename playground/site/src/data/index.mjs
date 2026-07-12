// CSV machinery translated from bayesledger src/ui/scratch-data.ts at commit 7346d71.

const CSV_DELIMITERS = [",", ";", "\t"];
const DATA_FORMAT = "bayescycle.data.json.v1";

export function sniffDelimiter(text) {
  const counts = new Map(CSV_DELIMITERS.map((delimiter) => [delimiter, 0]));
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index] ?? "";
    if (character === '"') {
      if (quoted && text[index + 1] === '"') index += 1;
      else quoted = !quoted;
    } else if (!quoted && (character === "\n" || character === "\r")) break;
    else if (!quoted && CSV_DELIMITERS.includes(character)) {
      counts.set(character, (counts.get(character) ?? 0) + 1);
    }
  }
  let selected = ",";
  let maximum = 0;
  let tied = false;
  for (const delimiter of CSV_DELIMITERS) {
    const count = counts.get(delimiter) ?? 0;
    if (count > maximum) {
      selected = delimiter;
      maximum = count;
      tied = false;
    } else if (count === maximum && count > 0) tied = true;
  }
  return maximum === 0 || tied ? "," : selected;
}

export function parseCsv(text, delimiter) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index] ?? "";
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else quoted = !quoted;
    } else if (character === delimiter && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else cell += character;
  }
  if (quoted) throw new Error("CSV has an unterminated quoted field");
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

/** Import CSV text into a plain typed-column store. */
export function importCsv(text) {
  const delimiter = sniffDelimiter(text);
  const parsed = parseCsv(text, delimiter);
  const header = parsed[0];
  if (header === undefined || header.length === 0) throw new Error("CSV needs a header row");

  const names = header.map((name, index) => identifier(name, index));
  const rows = parsed.slice(1).filter((row) => row.some((cell) => cell.length > 0));
  const columns = names.map((name, index) => typedColumn(name, rows.map((row) => row[index] ?? "")));
  const suspectedDelimiter = names.length === 1 ? findOtherDelimiter(parsed, delimiter) : undefined;

  return {
    delimiter,
    columns,
    rowCount: rows.length,
    ...(suspectedDelimiter === undefined ? {} : { suspectedDelimiter }),
  };
}

/** Parse and validate a bayescycle data document. */
export function importJson(text) {
  const document = JSON.parse(text);
  if (document === null || typeof document !== "object" || Array.isArray(document)) {
    throw new Error("data document must be an object");
  }
  if (document.format !== DATA_FORMAT) {
    throw new Error(`data document format must be ${DATA_FORMAT}`);
  }
  if (
    document.variables === null ||
    typeof document.variables !== "object" ||
    Array.isArray(document.variables)
  ) {
    throw new Error("data document variables must be an object");
  }

  for (const [key, entry] of Object.entries(document.variables)) validateVariable(key, entry);
  return document;
}

/** Return a standardized copy of a numeric column. */
export function standardize(column) {
  if (column.dtype === "string") throw new Error(`cannot standardize string column ${column.name}`);
  if (!column.values.every((value) => typeof value === "number")) {
    throw new Error(`column ${column.name} contains a non-numeric value`);
  }
  const mean = column.values.reduce((sum, value) => sum + value, 0) / column.values.length;
  const variance =
    column.values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / column.values.length;
  const sd = Math.sqrt(variance);
  if (sd === 0) throw new Error(`column ${column.name} has standard deviation 0`);
  return {
    name: column.name,
    dtype: "float64",
    standardized: true,
    values: column.values.map((value) => (value - mean) / sd),
  };
}

/** Extract declared and observed model inputs in declaration order. */
export function requiredInputs(ir) {
  const dataInputs = ir.model.data.map((entry) => {
    const schema = entry.value.schema;
    const dims = schema.dims?.map((dim) => dim.name) ?? [];
    return {
      name: entry.name,
      kind: dims.length > 0 || schema.rank > 0 ? "vector" : "scalar",
      dims,
    };
  });
  const dataNames = new Set(dataInputs.map((input) => input.name));
  const implicitRankSize =
    ir.model.data.some((entry) => entry.value.schema.rank > 0) && !dataNames.has("n")
      ? [{ name: "n", kind: "scalar", dims: [], synthetic: true }]
      : [];
  const observedInputs = ir.model.observed_nodes
    .filter((entry) => !dataNames.has(entry.name))
    .map((entry) => ({ name: entry.name, kind: "vector", dims: [] }));
  return [...dataInputs, ...implicitRankSize, ...observedInputs];
}

/** Bind typed columns to model inputs and produce an engine data document. */
export function bind(inputs, columns) {
  const vectorBindings = new Map();
  for (const input of inputs) {
    if (input.kind !== "vector") continue;
    const column = columns.find((candidate) => candidate.name === input.name);
    if (column !== undefined && column.dtype !== "string") vectorBindings.set(input.name, column);
  }

  let rowCount;
  let rowCountColumn;
  for (const [name, column] of vectorBindings) {
    if (column.values.some((value) => value === null)) {
      throw new Error(`column ${name} contains a null cell`);
    }
    if (rowCount === undefined) {
      rowCount = column.values.length;
      rowCountColumn = name;
    } else if (column.values.length !== rowCount) {
      throw new Error(
        `bound column length mismatch: ${name} has ${column.values.length} rows, ` +
          `${rowCountColumn} has ${rowCount}`,
      );
    }
  }

  const derivedScalars = new Set();
  for (const input of inputs) {
    if (input.kind === "vector" && vectorBindings.has(input.name)) {
      for (const dim of input.dims) derivedScalars.add(dim);
    }
  }

  const variables = {};
  const mapping = [];
  for (const input of inputs) {
    if (input.kind === "vector") {
      const column = columns.find((candidate) => candidate.name === input.name);
      if (column === undefined) {
        mapping.push({ input: input.name, source: null, status: "missing" });
      } else if (column.dtype === "string") {
        mapping.push({ input: input.name, source: null, status: "incompatible" });
      } else {
        variables[input.name] = {
          dtype: column.dtype,
          shape: [column.values.length],
          values: column.values,
        };
        mapping.push({ input: input.name, source: `column:${column.name}`, status: "bound" });
      }
    } else if (derivedScalars.has(input.name)) {
      variables[input.name] = { dtype: "int64", shape: [], values: [rowCount] };
      mapping.push({ input: input.name, source: "auto:length", status: "bound" });
    } else if (input.synthetic === true && rowCount !== undefined) {
      mapping.push({ input: input.name, source: "auto:length", status: "bound" });
    } else {
      mapping.push({ input: input.name, source: null, status: "missing" });
    }
  }

  return {
    document: { format: DATA_FORMAT, variables },
    mapping,
    complete: mapping.every((entry) => entry.status === "bound"),
  };
}

function findOtherDelimiter(rows, delimiter) {
  return CSV_DELIMITERS.find(
    (candidate) =>
      candidate !== delimiter && rows.some((row) => row.some((cell) => cell.includes(candidate))),
  );
}

function identifier(value, index) {
  const cleaned = value.trim().replace(/[^A-Za-z0-9_]/gu, "_");
  return cleaned.length === 0 ? `column_${String(index + 1)}` : cleaned;
}

function scalar(value) {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : value;
}

function typedColumn(name, cells) {
  const values = cells.map(scalar);
  const present = values.filter((value) => value !== null);
  if (present.every((value) => typeof value === "number" && Number.isInteger(value))) {
    return { name, dtype: "int64", values };
  }
  if (present.every((value) => typeof value === "number")) {
    return { name, dtype: "float64", values };
  }
  return {
    name,
    dtype: "string",
    values: cells.map((value) => {
      const trimmed = value.trim();
      return trimmed === "" ? null : trimmed;
    }),
  };
}

function validateVariable(key, entry) {
  if (entry === null || typeof entry !== "object" || Array.isArray(entry)) {
    throw new Error(`variable ${key} must be a data-document entry`);
  }
  if (entry.dtype !== "int64" && entry.dtype !== "float64") {
    throw new Error(`variable ${key} has invalid dtype`);
  }
  if (
    !Array.isArray(entry.shape) ||
    !entry.shape.every((size) => Number.isInteger(size) && size >= 0)
  ) {
    throw new Error(`variable ${key} has invalid shape`);
  }
  if (!Array.isArray(entry.values)) throw new Error(`variable ${key} values must be an array`);
  const expectedLength = entry.shape.reduce((product, size) => product * size, 1);
  if (entry.values.length !== expectedLength) {
    throw new Error(
      `variable ${key} shape requires ${expectedLength} values, got ${entry.values.length}`,
    );
  }
}
