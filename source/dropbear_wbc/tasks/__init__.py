"""Isaac Lab tasks for Dropbear. Call :func:`register` after the Isaac Sim app has started."""


def register() -> list[str]:
    """Register the gym ids of all dropbear-wbc tasks; returns them.

    Tracking ids are registered first and unconditionally. The velocity-locomotion ids (added 2026-09-24,
    ``tasks/locomotion``) are registered after them; an import error there is reported but does not break the
    tracking tools that call this function (scripts that need the velocity task import its config package directly
    and fail loudly).
    """
    from .tracking.config.dropbear import TASK_IDS

    ids = list(TASK_IDS)
    try:
        from .locomotion.config.dropbear import TASK_IDS as VELOCITY_IDS

        ids += list(VELOCITY_IDS)
    except Exception as exc:  # noqa: BLE001
        print(f"[dropbear_wbc.tasks.register] WARNING: velocity tasks not registered: {type(exc).__name__}: {exc}",
              flush=True)
    return ids
