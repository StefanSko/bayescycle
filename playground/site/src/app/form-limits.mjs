export const MAX_PARAMETER_FORM_FIELDS = 200;
export const MAX_DESIGN_FORM_SLOTS = 50;

export function supportsParameterForms(schema) {
  return schema.parameters.length <= MAX_PARAMETER_FORM_FIELDS;
}

// Integer slots (dimension sizes, index vectors) have no safe synthesized
// default, so any non-float64 or non-vector slot keeps the whole design in
// the JSON escape hatch. The slot cap also bounds rendered cards.
export function supportsDesignForms(schema) {
  return schema.data.length <= MAX_DESIGN_FORM_SLOTS &&
    schema.data.every((slot) => slot.kind === "vector" && slot.dtype === "float64");
}
