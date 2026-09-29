/** Shared route timings and limits. Next route modules may export handlers only. */
export const OWNER_EVENT_POLL_INTERVAL_MS = 30_000;
export const SSE_HEARTBEAT_INTERVAL_MS = 25_000;
export const OWNER_EVENT_REPLAY_LIMIT = 1_000;
export const SESSION_EVENT_POLL_INTERVAL_MS = 5_000;
export const TERMINAL_POLL_INTERVAL_MS = 10_000;
export const STAGE_FRESHNESS_POLL_INTERVAL_MS = 5_000;
export const STAGE_FRESHNESS_HEARTBEAT_MS = 25_000;
export const STAGE_FRESHNESS_RETRY_MS = 3_000;
export const MAX_BATCH_SCENE_IDS = 200;
