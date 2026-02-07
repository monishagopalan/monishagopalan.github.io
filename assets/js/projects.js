document.addEventListener("DOMContentLoaded", function () {
    const projectContainers = document.querySelectorAll(".project-container");

    projectContainers.forEach((container) => {
        const header = container.querySelector(".project-header-section");
        const grid = container.querySelector(".project-details-grid"); // Select the grid

        if (!header || !grid) return;

        // Create toggle button
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "project-toggle-btn";
        toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>'; // FontAwesome icon
        toggleBtn.title = "Collapse Project";

        // Find title and append button to it
        const title = container.querySelector(".project-main-title");
        if (title) {
            title.style.display = "flex";
            title.style.alignItems = "center";
            title.appendChild(toggleBtn);
        } else {
            header.appendChild(toggleBtn);
        }

        // Toggle function
        toggleBtn.addEventListener("click", () => {
            const isCollapsed = grid.style.display === "none";

            if (isCollapsed) {
                grid.style.display = "grid"; // Restore grid layout
                toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
                toggleBtn.title = "Collapse Project";
                toggleBtn.classList.remove("collapsed");
            } else {
                grid.style.display = "none";
                toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
                toggleBtn.title = "Expand Project";
                toggleBtn.classList.add("collapsed");
            }
        });
    });
});
