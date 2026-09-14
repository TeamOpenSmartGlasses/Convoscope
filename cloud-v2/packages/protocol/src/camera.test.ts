import { describe, expect, test } from "bun:test";

import {
  normalizePhotoSizeTier,
  photoOptionsSchema,
} from "./camera";

describe("normalizePhotoSizeTier", () => {
  test.each([
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["max", "max"],
    ["small", "low"],
    ["large", "high"],
    ["full", "max"],
  ] as const)("maps %s to %s", (input, expected) => {
    expect(normalizePhotoSizeTier(input)).toBe(expected);
  });

  test("rejects unknown values", () => {
    expect(() => normalizePhotoSizeTier("gigantic")).toThrow(/invalid photo size/);
  });
});

describe("photoOptionsSchema", () => {
  test.each([
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["max", "max"],
    ["small", "low"],
    ["large", "high"],
    ["full", "max"],
  ] as const)("accepts size %s and normalizes to %s", (input, expected) => {
    const parsed = photoOptionsSchema.parse({ size: input });
    expect(parsed.size).toBe(expected);
  });

  test("accepts omitted size", () => {
    expect(photoOptionsSchema.parse({})).toEqual({ compress: "none" });
  });

  test("rejects invalid size", () => {
    const result = photoOptionsSchema.safeParse({ size: "gigantic" });
    expect(result.success).toBe(false);
  });

  test.each(["none", "low", "medium", "high"])("preserves compression %s", (compress) => {
    expect(photoOptionsSchema.parse({ compress }).compress).toBe(compress);
  });

  test.each(["heavy", "LOW", "", "unknown", null, 1, false])("rejects compression %p", (compress) => {
    expect(photoOptionsSchema.safeParse({ compress }).success).toBe(false);
  });
});
