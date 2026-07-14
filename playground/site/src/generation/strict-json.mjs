export function parseStrictJson(text, label, options = {}) {
  const integerKeys = new Set(options.integerKeys ?? []);
  const integerArrayKeys = new Set(options.integerArrayKeys ?? []);
  const unrestrictedObjectKeys = new Set(options.unrestrictedObjectKeys ?? []);
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
  const integerToken = (token) => /^-?(?:0|[1-9]\d*)$/u.test(token);
  const value = (key = null, integerArray = false, unrestricted = false) => {
    whitespace();
    const character = text[index];
    if (character === "{") {
      objectValue(unrestricted || unrestrictedObjectKeys.has(key));
      return { value: undefined, allIntegerTokens: false };
    }
    if (character === "[") {
      const requireIntegers = integerArray || integerArrayKeys.has(key);
      let allIntegerTokens = true;
      index += 1;
      whitespace();
      if (text[index] === "]") {
        index += 1;
        return { value: undefined, allIntegerTokens: true };
      }
      while (true) {
        const parsed = value(null, requireIntegers, unrestricted);
        allIntegerTokens &&= parsed.allIntegerTokens;
        whitespace();
        if (text[index] === "]") {
          index += 1;
          return { value: undefined, allIntegerTokens };
        }
        if (text[index] !== ",") fail("has malformed array");
        index += 1;
      }
    }
    if (character === '"') {
      if (!unrestricted && (integerArray || integerKeys.has(key))) {
        fail(`field ${String(key)} must use an integer token`);
      }
      return { value: string(), allIntegerTokens: false };
    }
    const start = index;
    while (index < text.length && ![",", "]", "}", " ", "\t", "\r", "\n"].includes(text[index])) {
      index += 1;
    }
    const token = text.slice(start, index);
    const isIntegerToken = integerToken(token);
    if (!unrestricted && (integerArray || integerKeys.has(key)) && !isIntegerToken) {
      fail(`field ${String(key)} must use an integer token`);
    }
    return { value: token, allIntegerTokens: isIntegerToken };
  };
  const objectValue = (unrestricted = false) => {
    index += 1;
    const keys = new Set();
    let integerTypedValues = false;
    let valuesUseOnlyIntegerTokens = true;
    const finish = () => {
      if (integerTypedValues && !valuesUseOnlyIntegerTokens) {
        fail("integer-typed values must use integer tokens");
      }
    };
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
      const parsed = value(
        key, key === "values" && integerTypedValues, unrestricted,
      );
      if (key === "dtype") {
        integerTypedValues = parsed.value === "int32" || parsed.value === "int64";
      } else if (key === "values") {
        valuesUseOnlyIntegerTokens = parsed.allIntegerTokens;
      }
      whitespace();
      if (text[index] === "}") { index += 1; finish(); return; }
      if (text[index] !== ",") fail("has malformed object");
      index += 1;
    }
  };

  value();
  whitespace();
  if (index !== text.length) fail("has trailing JSON content");
  return JSON.parse(text);
}
