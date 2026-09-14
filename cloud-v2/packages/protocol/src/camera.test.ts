import { describe, expect, test } from "bun:test";

import {
  normalizePhotoCompress,
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
    expect(photoOptionsSchema.parse({})).toEqual({});
  });

  test("rejects invalid size", () => {
    const result = photoOptionsSchema.safeParse({ size: "gigantic" });
    expect(result.success).toBe(false);
  });

  test.each([
    ["none", "none"],
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["heavy", "high"],
  ] as const)("accepts compress %s and normalizes to %s", (input, expected) => {
    expect(photoOptionsSchema.parse({ compress: input }).compress).toBe(expected);
  });

  test("accepts omitted compress", () => {
    expect(photoOptionsSchema.parse({ size: "low" })).toEqual({ size: "low" });
  });

  test("rejects invalid compress", () => {
    expect(photoOptionsSchema.safeParse({ compress: "ultra" }).success).toBe(false);
    expect(photoOptionsSchema.safeParse({ compress: "" }).success).toBe(false);
  });
});

describe("normalizePhotoCompress", () => {
  test.each([
    ["none", "none"],
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["heavy", "high"],
  ] as const)("maps %s to %s", (input, expected) => {
    expect(normalizePhotoCompress(input)).toBe(expected);
  });

  test("keeps low and high as distinct tiers", () => {
    expect(normalizePhotoCompress("low")).not.toBe(normalizePhotoCompress("medium"));
    expect(normalizePhotoCompress("high")).not.toBe(normalizePhotoCompress("medium"));
  });

  test("never emits the legacy heavy spelling", () => {
    for (const input of ["none", "low", "medium", "high", "heavy"]) {
      expect(normalizePhotoCompress(input)).not.toBe("heavy");
    }
  });

  test("rejects unknown values", () => {
    expect(() => normalizePhotoCompress("ultra")).toThrow(/invalid photo compress/);
  });
});
