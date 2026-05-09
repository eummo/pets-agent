/**
 * ProjectStore — in-memory + file-persistent project state.
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import type { Project, Artifact, Decision } from "./types.js";
import { randomBytes } from "crypto";

const PROJECT_DIR = path.join(os.homedir(), ".pets-agent", "team");

export interface CreateProjectInput {
  name: string;
  description: string;
  target?: string;
  successCriteria?: string;
}

export class ProjectStore {
  private projects: Map<string, Project> = new Map();

  constructor() {
    this.load();
  }

  private filepath(): string {
    return path.join(PROJECT_DIR, "projects.json");
  }

  private load(): void {
    try {
      if (!fs.existsSync(PROJECT_DIR)) {
        fs.mkdirSync(PROJECT_DIR, { recursive: true });
        return;
      }
      const file = this.filepath();
      if (!fs.existsSync(file)) return;
      const raw = JSON.parse(fs.readFileSync(file, "utf-8")) as Project[];
      for (const p of raw) {
        this.projects.set(p.id, p);
      }
    } catch {
      // ignore
    }
  }

  private persist(): void {
    try {
      if (!fs.existsSync(PROJECT_DIR)) {
        fs.mkdirSync(PROJECT_DIR, { recursive: true });
      }
      const data = JSON.stringify(Array.from(this.projects.values()), null, 2);
      const tmp = this.filepath() + `.tmp.${randomBytes(4).toString("hex")}`;
      fs.writeFileSync(tmp, data, "utf-8");
      fs.renameSync(tmp, this.filepath());
    } catch (err) {
      console.error("[ProjectStore] persist failed:", err);
    }
  }

  create(input: CreateProjectInput): Project {
    const project: Project = {
      id: randomBytes(8).toString("hex"),
      name: input.name,
      description: input.description,
      target: input.target,
      successCriteria: input.successCriteria,
      phase: "idea",
      status: "planning",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      members: [],
      artifacts: [],
      decisions: [],
    };
    this.projects.set(project.id, project);
    this.persist();
    return project;
  }

  get(id: string): Project | undefined {
    return this.projects.get(id);
  }

  update(project: Project): void {
    project.updatedAt = new Date().toISOString();
    this.projects.set(project.id, project);
    this.persist();
  }

  addArtifact(projectId: string, artifact: Artifact): void {
    const p = this.projects.get(projectId);
    if (!p) return;
    p.artifacts.push(artifact);
    this.persist();
  }

  updateArtifact(projectId: string, artifact: Artifact): void {
    const p = this.projects.get(projectId);
    if (!p) return;
    const idx = p.artifacts.findIndex((a) => a.id === artifact.id);
    if (idx !== -1) {
      p.artifacts[idx] = artifact;
      this.persist();
    }
  }

  addDecision(projectId: string, decision: Decision): void {
    const p = this.projects.get(projectId);
    if (!p) return;
    p.decisions.push(decision);
    this.persist();
  }

  list(): Project[] {
    return Array.from(this.projects.values());
  }

  listActive(): Project[] {
    return this.list().filter((p) => p.status === "active" || p.status === "planning");
  }

  delete(id: string): void {
    this.projects.delete(id);
    this.persist();
  }
}

export const projectStore = new ProjectStore();
