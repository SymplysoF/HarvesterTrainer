(() => {
  const cfg = window.__AUG_PREVIEW;
  if (!cfg) return;

  const baseSrc =
    (cfg.projectId && cfg.sampleImageName)
      ? `/pimg/${cfg.projectId}/${encodeURIComponent(cfg.sampleImageName)}`
      : (cfg.fallbackSrc || "/static/augment_base.svg");

  // Set sources
  document.querySelectorAll("[data-aug-src]").forEach(el => {
    if (el.tagName === "IMG") el.src = baseSrc;
  });

  const map = [
    { key: "aug_color", thumb: "hsv" },
    { key: "aug_brightness", thumb: "bright" },
    { key: "aug_rotate", thumb: "rotate" },
    { key: "aug_flip", thumb: "flip" },
    { key: "aug_mosaic", thumb: "mosaic" },
  ];

  function sync() {
    map.forEach(({ key, thumb }) => {
      const cb = document.getElementById(key);
      const card = document.querySelector(`.thumb[data-thumb="${thumb}"]`);
      if (!cb || !card) return;
      card.classList.toggle("active", !!cb.checked);
      const tag = card.querySelector(".tag");
      if (tag) tag.textContent = cb.checked ? "выбрано" : "пример";
    });
  }

  // Click on preview toggles checkbox (nice UX)
	document.querySelectorAll(".thumb[data-thumb]").forEach(card => {
	  card.addEventListener("click", (e) => {
		const isInteractive = e.target.closest("a,button,input,select,textarea");
		if (isInteractive) return;

		const thumb = card.getAttribute("data-thumb");
		const entry = map.find(x => x.thumb === thumb);
		if (!entry) return;

		const cb = document.getElementById(entry.key);
		if (!cb) return;

		cb.checked = !cb.checked;
		cb.dispatchEvent(new Event("change", { bubbles: true }));
	  });
	});

  map.forEach(({ key }) => {
    const cb = document.getElementById(key);
    if (!cb) return;
    cb.addEventListener("change", sync);
  });

  sync();
})();