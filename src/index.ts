/**
 * Pets Agent - entry point
 *
 * Thin bootstrap: creates the app and starts the TUI.
 * All TUI logic lives in src/tui/
 */

import { createPetsApp, startPetsApp } from "./tui/app.js";

const app = createPetsApp();
startPetsApp(app);
