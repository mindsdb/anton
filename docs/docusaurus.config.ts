import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Anton',
  tagline: 'The open-source AI coworker',
  favicon: 'img/logo.svg',

  // GitHub Pages project site: https://mindsdb.github.io/anton/
  url: 'https://mindsdb.github.io',
  baseUrl: '/anton/',

  organizationName: 'mindsdb',
  projectName: 'anton',

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      /** @type {import("@easyops-cn/docusaurus-search-local").PluginOptions} */
      ({
        hashed: true,
        language: ['en'],
        indexBlog: false,
        docsRouteBasePath: '/',
        highlightSearchTermsOnTargetPage: false,
      }),
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // Docs at the site root
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/mindsdb/anton/edit/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/anton-diagram.png',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    navbar: {
      title: 'Anton',
      logo: {
        alt: 'Anton',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/mindsdb/anton',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Quickstart', to: '/start/quickstart'},
            {label: 'Use Anton', to: '/use/cli'},
            {label: 'Reference', to: '/reference/cli-commands'},
            {label: 'Under the Hood', to: '/developer/architecture'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub Issues', href: 'https://github.com/mindsdb/anton/issues'},
            {label: 'Slack Community', href: 'https://mindsdb.com/joincommunity'},
            {label: 'GitHub Discussions', href: 'https://github.com/mindsdb/mindsdb/discussions'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: 'https://github.com/mindsdb/anton'},
            {label: 'MindsDB', href: 'https://mindsdb.com'},
            {label: 'Minds Hub', href: 'https://mindshub.ai'},
          ],
        },
      ],
      copyright: `Built by <a href="https://mindsdb.com">MindsDB</a> · AGPL-3.0 License · ${new Date().getFullYear()}`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'python', 'toml', 'powershell'],
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
