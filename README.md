# Market Feed
📖 **Üzemeltetés:** a teljes operátori protokoll, döntési kapuk és hibaelhárítás: [docs/operator_handbook.md](docs/operator_handbook.md)


## Assumed position tracking
Manual/assumed positions are persisted in `trading.db`, and both the analysis
pipeline and Discord notifications derive the live `position_state` from that
database via `position_tracker.compute_state`. Opening or closing a position
updates the SQLite records atomically, so subsequent runs keep the same
`has_position` and `cooldown` view. To reset the state, delete the database or
remove the affected asset entry and the tracker will treat it as flat on the
next run.
