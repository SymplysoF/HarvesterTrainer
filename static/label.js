(() => {
  const cvs = document.getElementById("cvs");
  const ctx = cvs.getContext("2d");
  const classSelect = document.getElementById("classSelect");
  const btnSave = document.getElementById("btnSave");
  const saveStatus = document.getElementById("saveStatus");
  const boxesList = document.getElementById("boxesList");

  let img = new Image();
  img.src = imageUrl;
  let naturalW = 1, naturalH = 1;

  let boxes = (initialBoxes || []).map(b => ({
    class_id: Number(b.class_id),
    x_center: Number(b.x_center),
    y_center: Number(b.y_center),
    width: Number(b.width),
    height: Number(b.height),
  }));

  let drag = null;
  let selectedIndex = -1;

  function fitCanvas() {
    const maxW = 960;
    const maxH = 640;
    const s = Math.min(maxW / naturalW, maxH / naturalH, 1);
    cvs.width = Math.round(naturalW * s);
    cvs.height = Math.round(naturalH * s);
  }

  function denormBox(b) {
    const xc = b.x_center * cvs.width;
    const yc = b.y_center * cvs.height;
    const w = b.width * cvs.width;
    const h = b.height * cvs.height;
    return { x1: xc - w/2, y1: yc - h/2, x2: xc + w/2, y2: yc + h/2 };
  }

  function normFromPixels(x1, y1, x2, y2) {
    const xx1 = Math.max(0, Math.min(cvs.width, x1));
    const yy1 = Math.max(0, Math.min(cvs.height, y1));
    const xx2 = Math.max(0, Math.min(cvs.width, x2));
    const yy2 = Math.max(0, Math.min(cvs.height, y2));
    const left = Math.min(xx1, xx2);
    const right = Math.max(xx1, xx2);
    const top = Math.min(yy1, yy2);
    const bottom = Math.max(yy1, yy2);
    const w = Math.max(1, right - left);
    const h = Math.max(1, bottom - top);
    return {
      x_center: (left + right) / 2 / cvs.width,
      y_center: (top + bottom) / 2 / cvs.height,
      width: w / cvs.width,
      height: h / cvs.height
    };
  }

  function draw() {
    ctx.clearRect(0,0,cvs.width,cvs.height);
    ctx.drawImage(img, 0, 0, cvs.width, cvs.height);

    boxes.forEach((b, i) => {
      const r = denormBox(b);
      ctx.lineWidth = (i === selectedIndex) ? 3 : 2;
      ctx.strokeStyle = (i === selectedIndex) ? "#5aa9ff" : "#e9eef5";
      ctx.strokeRect(r.x1, r.y1, r.x2-r.x1, r.y2-r.y1);

      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(r.x1, Math.max(0, r.y1-18), 160, 18);
      ctx.fillStyle = "#e9eef5";
      ctx.font = "12px system-ui";
      ctx.fillText(`cls=${b.class_id}`, r.x1+6, Math.max(12, r.y1-6));
    });

    if (drag) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffcc66";
      const left = Math.min(drag.sx, drag.ex);
      const top = Math.min(drag.sy, drag.ey);
      const w = Math.abs(drag.ex - drag.sx);
      const h = Math.abs(drag.ey - drag.sy);
      ctx.strokeRect(left, top, w, h);
    }
  }

  function renderList() {
    boxesList.innerHTML = "";
    if (boxes.length === 0) {
      const div = document.createElement("div");
      div.className = "listitem";
      div.textContent = "Пока нет боксов";
      boxesList.appendChild(div);
      return;
    }
    boxes.forEach((b, i) => {
      const div = document.createElement("div");
      div.className = "listitem" + (i === selectedIndex ? " active" : "");
      div.style.cursor = "pointer";
      div.textContent = `#${i+1} class=${b.class_id} xc=${b.x_center.toFixed(3)} yc=${b.y_center.toFixed(3)} w=${b.width.toFixed(3)} h=${b.height.toFixed(3)}`;
      div.onclick = () => { selectedIndex = i; renderList(); draw(); };
      boxesList.appendChild(div);
    });
  }

  function canvasPos(evt) {
    const r = cvs.getBoundingClientRect();
    return {
      x: (evt.clientX - r.left) * (cvs.width / r.width),
      y: (evt.clientY - r.top) * (cvs.height / r.height)
    };
  }

  cvs.addEventListener("mousedown", (e) => {
    const p = canvasPos(e);

    selectedIndex = -1;
    for (let i=boxes.length-1; i>=0; i--) {
      const r = denormBox(boxes[i]);
      if (p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2) {
        selectedIndex = i;
        renderList();
        draw();
        return;
      }
    }

    drag = { sx: p.x, sy: p.y, ex: p.x, ey: p.y };
    draw();
  });

  cvs.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const p = canvasPos(e);
    drag.ex = p.x;
    drag.ey = p.y;
    draw();
  });

  cvs.addEventListener("mouseup", () => {
    if (!drag) return;
    const {sx,sy,ex,ey} = drag;
    drag = null;

    const w = Math.abs(ex - sx);
    const h = Math.abs(ey - sy);
    if (w < 6 || h < 6) { draw(); return; }

    const cls = Number(classSelect.value || 0);
    const n = normFromPixels(sx, sy, ex, ey);
    boxes.push({ class_id: cls, ...n });
    selectedIndex = boxes.length - 1;
    renderList();
    draw();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Delete" && selectedIndex >= 0) {
      boxes.splice(selectedIndex, 1);
      selectedIndex = Math.min(selectedIndex, boxes.length-1);
      renderList();
      draw();
    }
  });

  btnSave.addEventListener("click", async () => {
    saveStatus.textContent = "Сохраняю...";
    try {
      const res = await fetch(`/projects/${projectId}/label/save`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ img: imageName, boxes })
      });
      const js = await res.json();
      saveStatus.textContent = js.ok ? "Сохранено ✅" : ("Ошибка: " + (js.error || "unknown"));
    } catch (err) {
      saveStatus.textContent = "Ошибка: " + err;
    }
  });

  img.onload = () => {
    naturalW = img.naturalWidth || 1;
    naturalH = img.naturalHeight || 1;
    fitCanvas();
    renderList();
    draw();
  };
})();
