def _overlaps(start_a, end_a, start_b, end_b):
    return max(float(start_a), float(start_b)) < min(float(end_a), float(end_b))


def _object_label(value):
    if isinstance(value, dict):
        return value.get("label")
    return value


class SceneIntervalBuilder:
    def __init__(self, interval_sec=60.0):
        self.interval_sec = max(5.0, float(interval_sec))

    def _track_view(self, track_id, payload, interval_start, interval_end):
        metadata = payload.get("metadata", {})
        segments = []
        for segment in payload.get("segments", []):
            if _overlaps(segment["start"], segment["end"], interval_start, interval_end):
                segments.append(segment)

        return {
            "track_id": int(track_id),
            "first_seen": metadata.get("first_seen"),
            "last_seen": metadata.get("last_seen"),
            "objects": list(metadata.get("objects", [])),
            "object_tracks": list(metadata.get("object_tracks", [])),
            "interactions": list(metadata.get("interactions", [])),
            "segments": segments,
        }

    def build(self, event_log, track_payload):
        if not event_log:
            return []

        overall_start = min(window["start"] for window in event_log)
        overall_end = max(window["end"] for window in event_log)

        intervals = []
        cursor = float(overall_start)
        while cursor < overall_end:
            interval_end = cursor + self.interval_sec
            windows = [
                window
                for window in event_log
                if _overlaps(window["start"], window["end"], cursor, interval_end)
            ]
            active_tracks = sorted(
                {track_id for window in windows for track_id in window.get("active_tracks", [])}
            )
            events = [event for window in windows for event in window.get("events", [])]
            objects = sorted(
                {
                    _object_label(obj)
                    for event in events
                    if event.get("type") == "attributed_objects"
                    for obj in event.get("objects", [])
                    if _object_label(obj)
                }
            )
            interactions = [
                event for event in events if event.get("type") == "interaction"
            ]

            tracks = {}
            for track_id in active_tracks:
                payload = track_payload.get(track_id, {})
                tracks[track_id] = self._track_view(track_id, payload, cursor, interval_end)

            intervals.append(
                {
                    "start": round(cursor, 2),
                    "end": round(interval_end, 2),
                    "active_tracks": active_tracks,
                    "objects": objects,
                    "interaction_count": len(interactions),
                    "event_count": len(events),
                    "events": events,
                    "tracks": tracks,
                }
            )
            cursor = interval_end

        return intervals
