# Engineering Rules for Timed Technical Assessment

## Objective

Build the highest-quality solution realistically achievable within the available time.

Optimize for:

1. Correctness
2. Complete required functionality
3. Evidence that the solution works
4. Successful submission
5. Clear, maintainable architecture
6. Useful testing
7. Sensible polish

Architecture and code quality matter, but improvements must be proportional to their value and must never endanger correctness, completion, or submission.

## Scope

- Implement all explicit requirements.
- Do not invent requirements.
- Separate required work from optional polish.
- Prefer the smallest complete solution first.
- Do not implement speculative future functionality.
- Do not overengineer.

## Architecture

Prefer:

- simple, explicit architecture
- small understandable modules
- clear boundaries
- straightforward control flow
- descriptive names
- useful type hints
- easy-to-test components
- minimal dependencies

Avoid unless clearly justified:

- unnecessary abstraction layers
- factories or interfaces with only one implementation
- premature extensibility
- unnecessary databases or services
- clever code that is difficult to explain
- large rewrites of working code late in the task

Refactor when it meaningfully improves correctness, clarity, maintainability, or testability.

## Testing Philosophy

Prefer high-level behavioural tests first.

Tests should primarily verify externally observable behaviour and actual requirements.

Examples:

- API -> exercise the application/API boundary
- CLI -> exercise command behaviour and outputs
- data transformation -> realistic input to output
- workflow/agent -> exercise orchestration at the meaningful boundary

Add lower-level unit tests when they provide real value for:

- non-trivial algorithms
- parsing
- validation
- transformations
- branching business rules
- important edge cases

Do not:

- optimize for test count
- create trivial tests solely to increase coverage
- test implementation details unnecessarily
- mock so heavily that tests stop proving the real system works

Every important required behaviour should have a validation path.

Prefer a small number of meaningful tests over many low-value tests.

## Implementation

Before editing:

1. Inspect the relevant existing files.
2. Understand the current structure.
3. Preserve useful existing conventions.

While implementing:

- make coherent, scoped changes
- avoid unrelated edits
- keep behaviour deterministic where practical
- handle realistic failure modes
- validate inputs where appropriate
- keep code readable enough for another engineer to explain quickly

Do not rewrite working components unless there is a concrete reason.

## Dependencies

- Use only dependencies that provide clear value.
- Prefer standard library functionality when equally clear and practical.
- Do not add frameworks merely because they are familiar.
- Preserve dependency/version conventions in starter repositories.
- Do not introduce infrastructure that the requirements do not need.

## Error Handling

Handle realistic failures deliberately.

Prefer:

- meaningful errors
- appropriate status/exit codes
- graceful handling of bad input
- clear failure behaviour

Avoid swallowing exceptions without a reason.

Do not add elaborate retry/fallback machinery unless justified by the task.

## Security and Secrets

Never hardcode credentials or API keys.

Use environment variables or the configuration mechanism explicitly required by the task.

Ensure secret-bearing files such as `.env` are ignored by Git.

`.env.example` may contain placeholders only.

Do not print secrets in logs, tests, terminal output, or documentation.

Before final submission, verify that no credentials are tracked.

## Validation

After a meaningful change:

1. Run the smallest useful validation first.
2. Fix failures before expanding scope.
3. Run broader relevant tests once the feature works.

Before final submission, where available and time permits:

- run the full relevant test suite
- run configured linting
- run configured type/static checks
- manually exercise the main happy path
- verify at least one important failure/edge case

Do not spend critical submission time fixing cosmetic lint issues that do not affect correctness unless explicitly required.

## Codex Working Style

When given a task:

1. Inspect the repository before modifying it.
2. Implement only the requested scope.
3. Prefer the simplest correct approach.
4. Add meaningful tests where appropriate.
5. Run relevant validation.
6. Report what changed.

At the end of each task, report concisely:

- files changed
- purpose of each meaningful change
- tests/checks run
- their results
- remaining limitations or incomplete requirements
- 2-5 most important files/functions the developer should inspect

Do not silently continue into optional work.

## Time-Constrained Behaviour

When told that time is low:

### Under 15 minutes

- prioritize required functionality
- stop speculative enhancements
- fix correctness blockers
- preserve a working state
- prepare for submission

### Under 7 minutes — SUBMISSION MODE

Do not:

- perform major refactors
- introduce new architecture
- add optional features

Only:

- fix submission/correctness blockers
- run essential validation
- check secrets
- finalize documentation if required
- commit/push/package/submit

A working, tested, submitted solution is better than an unfinished architectural improvement.