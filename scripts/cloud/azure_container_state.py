"""Read ACI provisioning responses, which can precede runtime instance views."""


def current_state(resource):
    containers = resource.get('containers') or []
    if not containers:
        return {}
    return (containers[0].get('instanceView') or {}).get('currentState') or {}
