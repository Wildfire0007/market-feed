# Market Feed

## Assumed position tracking
Manual/assumed positions are persisted in `public/_manual_positions.json`, and
both the analysis pipeline and Discord notifications derive the live
`position_state` from that file via `position_tracker.compute_state`. Opening or
closing a position updates the JSON atomically, so subsequent runs keep the
same `has_position` and `cooldown` view. To reset the state, delete the file or
remove the affected asset entry and the tracker will treat it as flat on the
next run.
