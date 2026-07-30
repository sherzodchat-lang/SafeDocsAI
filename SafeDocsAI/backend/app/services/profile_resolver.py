from app.domain_profiles import get_domain_profile
from app.shared.models import Notebook
from app.shared.settings import RuntimeSettingsService


def resolve_profile_name(
    notebook: Notebook | None = None, requested: str | None = None
) -> str:
    if requested:
        return get_domain_profile(requested).name
    if notebook and notebook.domain_profile:
        return get_domain_profile(notebook.domain_profile).name
    runtime_settings = RuntimeSettingsService.get_settings()
    return get_domain_profile(runtime_settings.get("default_domain_profile")).name


def resolve_profile(notebook: Notebook | None = None, requested: str | None = None):
    return get_domain_profile(
        resolve_profile_name(notebook=notebook, requested=requested)
    )
