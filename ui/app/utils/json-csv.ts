//Function to convert json to csv
export const jsonToCsv = (json: Record<string, any>[]): string => {
  // Handle empty input
  if (!Array.isArray(json) || json.length === 0) {
    return "";
  }

  try {
    // Flatten the first object to get all possible headers
    const flattenedFirst = flattenObject(json[0] || {});
    const headers = Object.keys(flattenedFirst);

    // Create CSV header row
    const headerRow = headers.map((header) => escapeCsvValue(header)).join(",");

    // Convert data rows with flattened objects
    const rows = json.map((row) => {
      const flatRow = flattenObject(row);
      return headers.map((header) => escapeCsvValue(flatRow[header])).join(",");
    });

    return [headerRow, ...rows].join("\n");
  } catch (error) {
    console.error("Error converting JSON to CSV:", error);
    return "";
  }
};

// New helper function to flatten nested objects
const flattenObject = (
  obj: Record<string, any>,
  prefix = ""
): Record<string, any> => {
  return Object.keys(obj).reduce((acc: Record<string, any>, key: string) => {
    const prefixedKey = prefix ? `${prefix}.${key}` : key;

    if (obj[key] && typeof obj[key] === "object" && !Array.isArray(obj[key])) {
      Object.assign(acc, flattenObject(obj[key], prefixedKey));
    } else {
      acc[prefixedKey] = obj[key];
    }

    return acc;
  }, {});
};

// Helper function to properly escape CSV values
const escapeCsvValue = (value: any): string => {
  if (value === null || value === undefined) {
    return "";
  }

  const stringValue = String(value);

  // If value contains comma, newline, or double quote, wrap in quotes
  if (
    stringValue.includes(",") ||
    stringValue.includes("\n") ||
    stringValue.includes('"')
  ) {
    // Double up any existing quotes and wrap in quotes
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
};
