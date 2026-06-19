import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'index',
    {
      type: 'category',
      label: 'Get Started',
      items: [
        'start/quickstart',
        'start/install',
        'start/pick-a-provider',
        'start/updating',
      ],
    },
    {
      type: 'category',
      label: 'Use Anton',
      items: [
        'use/cli',
        'use/desktop',
        'use/chat-basics',
        'use/sessions',
        'use/workspaces',
        'use/dashboard',
      ],
    },
    {
      type: 'category',
      label: 'Connect Things',
      items: [
        'connect/overview',
        'connect/data-sources',
        'connect/web-search',
        'connect/web-fetch',
        'connect/custom-integrations',
      ],
    },
    {
      type: 'category',
      label: 'Teach Anton',
      items: [
        'teach/memory-overview',
        'teach/lessons-and-rules',
        'teach/skills',
        'teach/episodes-and-recall',
        'teach/project-context',
        'teach/learnings-cli',
      ],
    },
    {
      type: 'category',
      label: 'Configure',
      items: [
        'configure/env-vars',
        'configure/analytics',
        'configure/trace-headers',
        'configure/search-providers',
        'configure/security',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli-commands',
        'reference/slash-commands',
        'reference/workspace-files',
        'reference/glossary',
      ],
    },
    {
      type: 'category',
      label: 'Under the Hood',
      items: [
        'developer/architecture',
        'developer/brain-mapping',
        'developer/memory-systems',
        'developer/cerebellum-and-acc',
        'developer/skills-internals',
        'developer/scratchpad-runtime',
        'developer/llm-dispatch',
        'developer/tool-system',
        'developer/adding-a-datasource',
        'developer/adding-a-tool',
        'developer/release-and-versioning',
        'developer/contributing',
      ],
    },
  ],
};

export default sidebars;
