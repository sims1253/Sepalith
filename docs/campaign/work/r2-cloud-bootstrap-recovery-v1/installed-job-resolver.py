from __future__ import annotations
class InstalledPrivateJobSDK:
    def _resolve_to_job_model(
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> ProductionJob:
        if name is None and job_id is None:
            raise ValueError("One of 'name' or 'job_id' must be provided.")

        if name is not None and job_id is not None:
            raise ValueError("Only one of 'name' or 'job_id' can be provided.")

        if job_id is not None and (cloud is not None or project is not None):
            raise ValueError("'cloud' and 'project' should only be used with 'name'.")

        try:
            model: Optional[ProductionJob] = self.client.get_job(
                name=name,
                job_id=job_id,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
        except Exception as e:
            # Convert API exceptions to RuntimeError for user-friendly error messages
            if name is not None:
                raise RuntimeError(f"Job with name '{name}' was not found.") from e
            else:
                raise RuntimeError(f"Job with ID '{job_id}' was not found.") from e

        if model is None:
            if name is not None:
                raise RuntimeError(f"Job with name '{name}' was not found.")
            else:
                raise RuntimeError(f"Job with ID '{job_id}' was not found.")

        return model
    def status(
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> JobStatus:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )
        runs = self.client.get_job_runs(job_model.id)
        return self._job_model_to_status(model=job_model, runs=runs)
    def terminate(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )
        self.client.terminate_job(job_model.id)
        self.logger.info(f"Marked job '{job_model.name}' for termination")
        return job_model.id
