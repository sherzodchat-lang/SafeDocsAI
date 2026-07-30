from app.domain_profiles.base import DomainProfile


class GeneralDomainProfile(DomainProfile):
    def __init__(self) -> None:
        super().__init__(name="general", assistant_name="SafeDocsAI")
