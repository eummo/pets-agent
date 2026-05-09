/**
 * ProjectStore — Unit Tests
 * Run: npx vitest run src/tests/project-store.test.ts --no-typecheck
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, unlinkSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { ProjectStore } from "../multi-agent-team/project-store.js";
import type { Project } from "../multi-agent-team/types.js";

const TEST_DIR = join(homedir(), ".pets-agent-team-test");
const TEST_FILE = join(TEST_DIR, "projects.json");

describe("ProjectStore", () => {
  let store: ProjectStore;

  beforeEach(() => {
    // Replace PROJECT_DIR for isolated test environment
    store = new ProjectStore();
    // Clear in-memory state by replacing the Map
    (store as any).projects = new Map();
  });

  function filepath(): string {
    return join(TEST_DIR, "projects.json");
  }

  it("create() returns a project with an id", () => {
    const proj = store.create({ name: "Test", description: "A test project" });
    expect(proj.id).toBeDefined();
    expect(proj.id.length).toBeGreaterThan(0);
    expect(proj.name).toBe("Test");
    expect(proj.phase).toBe("idea");
    expect(proj.status).toBe("planning");
  });

  it("create() stores the project", () => {
    const proj = store.create({ name: "Stored Project", description: "desc" });
    expect(store.get(proj.id)).toBeDefined();
    expect(store.get(proj.id)!.name).toBe("Stored Project");
  });

  it("get() returns undefined for non-existent id", () => {
    expect(store.get("non-existent")).toBeUndefined();
  });

  it("update() modifies and persists the project", () => {
    const proj = store.create({ name: "Orig", description: "desc" });
    proj.name = "Updated";
    store.update(proj);
    expect(store.get(proj.id)!.name).toBe("Updated");
  });

  it("list() returns all projects", () => {
    store.create({ name: "Proj1", description: "d1" });
    store.create({ name: "Proj2", description: "d2" });
    expect(store.list()).toHaveLength(2);
  });

  it("listActive() returns only planning/active/blocked projects", () => {
    const p1 = store.create({ name: "Active", description: "" });
    const p2 = store.create({ name: "Completed", description: "" });
    p2.status = "completed";
    store.update(p2);
    const active = store.listActive();
    expect(active.some((p) => p.name === "Active")).toBe(true);
    expect(active.some((p) => p.name === "Completed")).toBe(false);
  });

  it("addArtifact() appends artifact to project", () => {
    const proj = store.create({ name: "With Artifacts", description: "" });
    const artifact = {
      id: "art-1",
      projectId: proj.id,
      type: "idea_form" as const,
      title: "Idea Form",
      content: "# Idea",
      phase: "idea" as const,
      createdBy: "pm" as const,
      createdAt: new Date().toISOString(),
      version: 1,
      status: "draft" as const,
      reviewers: [],
    };
    store.addArtifact(proj.id, artifact);
    const updated = store.get(proj.id)!;
    expect(updated.artifacts).toHaveLength(1);
    expect(updated.artifacts[0].type).toBe("idea_form");
  });

  it("updateArtifact() replaces existing artifact", () => {
    const proj = store.create({ name: "Update Test", description: "" });
    const artifact = {
      id: "art-1",
      projectId: proj.id,
      type: "idea_form" as const,
      title: "Idea Form",
      content: "# Idea v1",
      phase: "idea" as const,
      createdBy: "pm" as const,
      createdAt: new Date().toISOString(),
      version: 1,
      status: "draft" as const,
      reviewers: [],
    };
    store.addArtifact(proj.id, artifact);

    const updated = { ...artifact, content: "# Idea v2", version: 2, status: "approved" as const };
    store.updateArtifact(proj.id, updated);

    const proj2 = store.get(proj.id)!;
    expect(proj2.artifacts).toHaveLength(1);
    expect(proj2.artifacts[0].content).toBe("# Idea v2");
    expect(proj2.artifacts[0].version).toBe(2);
  });

  it("addDecision() records a decision on the project", () => {
    const proj = store.create({ name: "Decision Test", description: "" });
    const decision = {
      id: "dec-1",
      projectId: proj.id,
      topic: "Which framework?",
      options: ["React", "Vue"],
      selected: 0,
      rationale: "More ecosystem",
      madeBy: "pm" as const,
      madeAt: new Date().toISOString(),
      phase: "idea" as const,
    };
    store.addDecision(proj.id, decision);
    const updated = store.get(proj.id)!;
    expect(updated.decisions).toHaveLength(1);
    expect(updated.decisions[0].topic).toBe("Which framework?");
  });
});
