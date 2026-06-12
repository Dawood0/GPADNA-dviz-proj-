(() => {
    let draggedChip = null;

    function updateSelectedFeatures(board) {
        const selected = [...board.querySelectorAll('[data-zone="selected"] .feature-chip')]
            .map((chip) => chip.dataset.feature);

        if (window.dash_clientside?.set_props) {
            window.dash_clientside.set_props(board.dataset.storeId, {data: selected});
        }
    }

    function positionChip(zone, beforeChip = null) {
        if (!draggedChip || !zone) return;

        zone.insertBefore(draggedChip, beforeChip);
        draggedChip.classList.toggle("selected", zone.dataset.zone === "selected");
    }

    document.addEventListener("dragstart", (event) => {
        const chip = event.target.closest(".feature-chip");
        if (!chip) return;

        draggedChip = chip;
        chip.classList.add("dragging");
        event.dataTransfer.effectAllowed = "move";
    });

    document.addEventListener("dragend", () => {
        draggedChip?.classList.remove("dragging");
        draggedChip = null;
        document.querySelectorAll(".feature-drop-zone").forEach((zone) => zone.classList.remove("drag-over"));
    });

    document.addEventListener("dragover", (event) => {
        const zone = event.target.closest(".feature-drop-zone");
        if (!zone) return;

        event.preventDefault();
        zone.classList.add("drag-over");
        const beforeChip = event.target.closest(".feature-chip");
        if (beforeChip !== draggedChip) positionChip(zone, beforeChip);
    });

    document.addEventListener("dragleave", (event) => {
        const zone = event.target.closest(".feature-drop-zone");
        if (zone && !zone.contains(event.relatedTarget)) zone.classList.remove("drag-over");
    });

    document.addEventListener("drop", (event) => {
        const zone = event.target.closest(".feature-drop-zone");
        if (!zone) return;

        event.preventDefault();
        positionChip(zone, draggedChip.nextElementSibling);
        updateSelectedFeatures(zone.closest(".feature-drag-board"));
        zone.classList.remove("drag-over");
    });
})();
