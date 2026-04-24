(() => {
  function setActiveNav() {
    const path = window.location.pathname || "/";
    const links = document.querySelectorAll(".nav a[data-nav]");
    links.forEach(a => {
      const base = a.getAttribute("data-nav") || "";
      const isActive = (base === "/" && path === "/") || (base !== "/" && path.startsWith(base));
      a.classList.toggle("active", isActive);
    });
  }

  function initTableFilters() {
    document.querySelectorAll("input[data-filter-table]").forEach(inp => {
      const tableId = inp.getAttribute("data-filter-table");
      const table = document.getElementById(tableId);
      if (!table) return;
      const rows = Array.from(table.querySelectorAll("tbody tr"));

      const apply = () => {
        const q = (inp.value || "").trim().toLowerCase();
        rows.forEach(tr => {
          const txt = (tr.textContent || "").toLowerCase();
          tr.style.display = (!q || txt.includes(q)) ? "" : "none";
        });
      };

      inp.addEventListener("input", apply);
      apply();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    setActiveNav();
    initTableFilters();
  });
})();