(function () {
  'use strict';
  const resolution = document.getElementById('budget-resolution');
  if (!resolution) return;
  const equation = document.getElementById('budget-equation');
  const verdict = document.getElementById('budget-verdict');
  function update() {
    const side = Number(resolution.value);
    const axons = side * side * 2;
    const limit = SNN.AXON_LIMITS.total_axons;
    equation.textContent = `${side} × ${side} × 2 = ${axons.toLocaleString('en-US')} input axons`;
    verdict.textContent = axons > limit
      ? `Exceeds the ${limit.toLocaleString('en-US')}-axon limit by ${(axons - limit).toLocaleString('en-US')}. Resize before training.`
      : `Input fits with ${(limit - axons).toLocaleString('en-US')} axons remaining. Layer connection checks still apply.`;
  }
  resolution.addEventListener('change', update);
  update();
})();
