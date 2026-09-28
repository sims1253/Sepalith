# Campaign board design

The audience is the Sepalith researcher and the agents helping deliver Tuesday's R editing model. The board's job is to expose the next executable task, its prerequisites, and the evidence needed to finish it.

Palette: sky `#DDF1FF`, white `#FFFFFF`, ink `#17314D`, cobalt `#2453D4`, mint `#C1E9D5`, amber `#FFD287`. Blue is the working surface; mint identifies completion and amber identifies an unresolved dependency. No dark theme or cream paper treatment.

Type: Barlow Semi Condensed for the campaign heading and phase names; Public Sans for task content and controls. System sans fallbacks keep the hosted copy usable under a restrictive content policy. Body lines stay below 80 characters.

Layout: the experiment sequence leads the page. A compact phase rail sits beside task rows, with the selected task's executable brief in a separate right column. On a small screen, the phase rail becomes a horizontal selector and the brief follows its task. All working content is left aligned.

```text
Sepalith                                     Guide / Export
A better edit, by Tuesday.    Prompt -- SFT -- RL -- Release
----------------------------------------------------------
Phases        Search / Ready / Required      Task brief
Preparation   Task rows with dependencies   Steps
Prompt        Task rows with owners         Acceptance
Data          Task rows with resource       Receipt / notes
...                                         Delegate task
----------------------------------------------------------
Resource limits and release cutoffs
```

Review: a generic dashboard would lead with four metric cards and treat every task as a floating card. Replace that with the actual model-training sequence and grouped task rows. Spend visual emphasis on the cobalt sequence connector; keep working controls quiet. Counts describe recorded checklist state, never live GPU activity.

The local tracker and CLI share `state.json`. A file-opened tracker stores progress in its browser and offers export/import. PostPlan's published snapshot uses native links and expandable details because its documented CSP disables JavaScript. It must display that limitation rather than showing controls that cannot save.

Keyboard focus is visible. Status always has text as well as color. Motion is limited to user-triggered disclosure and honors reduced motion. Review desktop and mobile screenshots, as well as dependency and persistence behavior, before handoff.
