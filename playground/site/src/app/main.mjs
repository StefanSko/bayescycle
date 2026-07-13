// Browser UI adapter. Runtime behavior lands in later red→green work orders.
const status = document.querySelector("#compile-status");
document.querySelector("#compile-button").addEventListener("click", () => {
  status.textContent = "Compiler runtime is not implemented yet.";
});
