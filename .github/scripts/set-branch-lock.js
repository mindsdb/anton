function enabled(setting, fallback = false) {
  return typeof setting?.enabled === 'boolean' ? setting.enabled : fallback;
}

function names(items, key) {
  return (items ?? []).map((item) => item[key]).filter(Boolean);
}

function appSlugs(items) {
  return names(items, 'slug');
}

function restrictions(restriction) {
  if (!restriction) {
    return null;
  }

  return {
    users: names(restriction.users, 'login'),
    teams: names(restriction.teams, 'slug'),
    apps: appSlugs(restriction.apps),
  };
}

function reviewRules(reviews) {
  if (!reviews) {
    return null;
  }

  const rules = {
    required_approving_review_count:
      reviews.required_approving_review_count ?? 1,
  };

  if (reviews.dismissal_restrictions) {
    rules.dismissal_restrictions = restrictions(reviews.dismissal_restrictions);
  }

  if (typeof reviews.dismiss_stale_reviews === 'boolean') {
    rules.dismiss_stale_reviews = reviews.dismiss_stale_reviews;
  }

  if (typeof reviews.require_code_owner_reviews === 'boolean') {
    rules.require_code_owner_reviews = reviews.require_code_owner_reviews;
  }

  if (typeof reviews.require_last_push_approval === 'boolean') {
    rules.require_last_push_approval = reviews.require_last_push_approval;
  }

  if (reviews.bypass_pull_request_allowances) {
    rules.bypass_pull_request_allowances = restrictions(
      reviews.bypass_pull_request_allowances
    );
  }

  return rules;
}

function statusChecks(checks) {
  if (!checks) {
    return null;
  }

  const rules = {
    strict: checks.strict ?? false,
    contexts: checks.contexts ?? [],
  };

  if (checks.checks) {
    rules.checks = checks.checks.map((check) => {
      const rule = { context: check.context };
      if (typeof check.app_id === 'number') {
        rule.app_id = check.app_id;
      }
      return rule;
    });
  }

  return rules;
}

module.exports = async function setBranchLock({
  github,
  context,
  core,
  branch,
  locked,
}) {
  let current = null;
  try {
    current = await github.rest.repos.getBranchProtection({
      owner: context.repo.owner,
      repo: context.repo.repo,
      branch,
    });
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }
  }

  const protection = current?.data;

  await github.rest.repos.updateBranchProtection({
    owner: context.repo.owner,
    repo: context.repo.repo,
    branch,
    required_status_checks: statusChecks(protection?.required_status_checks),
    enforce_admins: protection?.enforce_admins?.enabled ?? true,
    required_pull_request_reviews: reviewRules(
      protection?.required_pull_request_reviews
    ),
    restrictions: restrictions(protection?.restrictions),
    required_linear_history: enabled(protection?.required_linear_history),
    allow_force_pushes: enabled(protection?.allow_force_pushes),
    allow_deletions: enabled(protection?.allow_deletions),
    block_creations: enabled(protection?.block_creations),
    required_conversation_resolution: enabled(
      protection?.required_conversation_resolution
    ),
    lock_branch: locked,
    allow_fork_syncing: enabled(protection?.allow_fork_syncing),
  });

  core.info(`${branch} branch ${locked ? 'locked' : 'unlocked'}`);
};
