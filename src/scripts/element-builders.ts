/**
 * Create a new table to hold the results
 *
 * @param id The id that the table should hold.
 */
export function createResultTable(id: string): HTMLTableElement {
	// Make the headers
	const resultTable = document.createElement("table");
	resultTable.id = id;
	resultTable.classList.add("table");
	resultTable.classList.add("table", "table-striped", "table-responsive", "mt-4");
	const header = resultTable.createTHead();
	const headerRow = header.insertRow();
	for (const entry of [
		{ text: "Iterations", tooltip: "Number of iterations" },
		{
			text: "Shader time (ms)",
			tooltip: "Amount of time passed measured using timestamp queries",
		},
		{ text: "Js time (ms)", tooltip: "" },
		{
			text: "Result",
			tooltip: "The computed value of some random element in the result matrix",
		},
	]) {
		const th = document.createElement("th");
		th.setAttribute("scope", "col");
		th.textContent = entry.text;
		th.title = entry.tooltip;
		headerRow.appendChild(th);
	}
	const tbody = resultTable.createTBody();
	tbody.id = `${id}-body`;
	return resultTable;
}
