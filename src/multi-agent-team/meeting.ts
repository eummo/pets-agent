/**
 * Meeting — meeting and decision record management.
 */

import type { Meeting, Decision, TeamRole, ProjectPhase } from "./types.js";
import { randomBytes } from "crypto";

export interface CreateMeetingInput {
  projectId: string;
  topic: string;
  participants: TeamRole[];
  notes?: string;
}

export interface CreateDecisionInput {
  projectId: string;
  topic: string;
  options: string[];
  rationale: string;
  selected: number;
  madeBy: TeamRole;
  phase: ProjectPhase;
}

export class MeetingManager {
  private meetings: Map<string, Meeting> = new Map();
  private decisions: Map<string, Decision> = new Map();

  createMeeting(input: CreateMeetingInput): Meeting {
    const meeting: Meeting = {
      id: randomBytes(4).toString("hex"),
      projectId: input.projectId,
      topic: input.topic,
      participants: input.participants,
      notes: input.notes ?? "",
      outcomes: [],
      createdAt: new Date().toISOString(),
    };
    this.meetings.set(meeting.id, meeting);
    return meeting;
  }

  addMeetingOutcome(meetingId: string, outcome: string): Meeting | null {
    const meeting = this.meetings.get(meetingId);
    if (!meeting) return null;
    meeting.outcomes.push(outcome);
    return meeting;
  }

  createDecision(input: CreateDecisionInput): Decision {
    const decision: Decision = {
      id: randomBytes(4).toString("hex"),
      projectId: input.projectId,
      topic: input.topic,
      options: input.options,
      selected: input.selected,
      rationale: input.rationale,
      madeBy: input.madeBy,
      phase: input.phase,
      madeAt: new Date().toISOString(),
    };
    this.decisions.set(decision.id, decision);
    return decision;
  }

  getDecisionsForProject(projectId: string): Decision[] {
    return Array.from(this.decisions.values()).filter((d) => d.projectId === projectId);
  }

  getMeetingsForProject(projectId: string): Meeting[] {
    return Array.from(this.meetings.values()).filter((m) => m.projectId === projectId);
  }

  formatDecision(d: Decision): string {
    const lines = [
      `## 决策: ${d.topic}`,
      `时间: ${new Date(d.madeAt).toLocaleString("zh-CN")}`,
      `决策者: ${d.madeBy}`,
      `阶段: ${d.phase}`,
      "",
      "选项:",
      ...d.options.map((o, i) => `  ${i + 1}. ${o}`),
      "",
      `✓ 选择: ${d.options[d.selected]}`,
      "",
      `理由: ${d.rationale}`,
    ];
    return lines.join("\n");
  }

  formatMeeting(m: Meeting): string {
    const lines = [
      `## 会议: ${m.topic}`,
      `时间: ${new Date(m.createdAt).toLocaleString("zh-CN")}`,
      `参与者: ${m.participants.join(", ")}`,
      "",
      m.notes || "(无记录)",
      "",
      ...(m.outcomes.length > 0
        ? ["决议:", ...m.outcomes.map((o) => `  • ${o}`)]
        : []),
    ];
    return lines.join("\n");
  }
}

export const meetingManager = new MeetingManager();
