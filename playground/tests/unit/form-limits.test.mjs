import {
  MAX_DESIGN_FORM_SLOTS,
  MAX_PARAMETER_FORM_FIELDS,
  supportsDesignForms,
  supportsParameterForms,
} from "/site/src/app/form-limits.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

const parameter = (index) => ({ name: `p${index}` });
const slot = (index) => ({ name: `x${index}`, kind: "vector", dtype: "float64" });

export default [
  {
    name: "parameter forms stop above two hundred fields",
    fn: () => {
      assert(supportsParameterForms({
        parameters: Array.from({ length: MAX_PARAMETER_FORM_FIELDS }, (_, index) => parameter(index)),
      }), "parameter form rejected its boundary");
      assert(!supportsParameterForms({
        parameters: Array.from({ length: MAX_PARAMETER_FORM_FIELDS + 1 }, (_, index) => parameter(index)),
      }), "parameter form accepted an over-cap schema");
    },
  },
  {
    name: "design forms stop above fifty compatible slots",
    fn: () => {
      assert(supportsDesignForms({
        data: Array.from({ length: MAX_DESIGN_FORM_SLOTS }, (_, index) => slot(index)),
      }), "design form rejected its boundary");
      assert(!supportsDesignForms({
        data: Array.from({ length: MAX_DESIGN_FORM_SLOTS + 1 }, (_, index) => slot(index)),
      }), "design form accepted an over-cap schema");
    },
  },
];
