export function parseStrictJson(text, label, options = {}) {
  const integerKeys = new Set(options.integerKeys ?? []);
  const integerArrayKeys = new Set(options.integerArrayKeys ?? []);
  let index = 0;

  const fail = (message) => { throw new Error(`${label} ${message}`); };
  const whitespace = () => {
    while ([" ", "\t", "\r", "\n"].includes(text[index])) index += 1;
  };
  const string = () => {
    const start = index;
    index += 1;
    let escaped = false;
    while (index < text.length) {
      const character = text[index++];
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') return JSON.parse(text.slice(start, index));
    }
    fail("has malformed string");
  };
  const value = (key = null, integerArray = false) => {
    whitespace();
    const character = text[index];
    if (character === "{") {
      objectValue();
      return;
    }
    if (character === "[") {
      const requireIntegers = integerArray || integerArrayKeys.has(key);
      index += 1;
      whitespace();
      if (text[index] === "]") { index += 1; return; }
      while (true) {
        value(null, requireIntegers);
        whitespace();
        if (text[index] === "]") { index += 1; return; }
        if (text[index] !== ",") fail("has malformed array");
        index += 1;
      }
    }
    if (character === '"') {
      if (integerArray || integerKeys.has(key)) fail(`field ${String(key)} must use an integer token`);
      string();
      return;
    }
    const start = index;
    while (index < text.length && ![",", "]", "}", " ", "\t", "\r", "\n"].includes(text[index])) {
      index += 1;
    }
    const token = text.slice(start, index);
    if ((integerArray || integerKeys.has(key)) && !/^(?:0|-[1-9]\d*|[1-9]\d*)$/u.test(token)) {
      fail(`field ${String(key)} must use an integer token`);
    }
  };
  const objectValue = () => {
    index += 1;
    const keys = new Set();
    whitespace();
    if (text[index] === "}") { index += 1; return; }
    while (true) {
      whitespace();
      if (text[index] !== '"') fail("has malformed object key");
      const key = string();
      if (keys.has(key)) fail(`has duplicate object key ${key}`);
      keys.add(key);
      whitespace();
      if (text[index] !== ":") fail("has malformed object field");
      index += 1;
      value(key, false);
      whitespace();
      if (text[index] === "}") { index += 1; return; }
      if (text[index] !== ",") fail("has malformed object");
      index += 1;
    }
  };

  value();
  whitespace();
  if (index !== text.length) fail("has trailing JSON content");
  return JSON.parse(text);
}
