"""Bundled MindsDB policy texts shown inline by the CLI consent flow.

These constants mirror the canonical versions hosted at
https://mindsdb.com/terms and https://mindsdb.com/privacy-policy. Update
both this file and the hosted pages together so they stay in sync.

Module is imported lazily inside `_ensure_terms_consent` so the ~14KB of
string data does not load on every CLI invocation.
"""

from __future__ import annotations


TERMS_OF_USE_MD = """\
# MindsDB Digital Website Terms of Use

*Last Updated: September 24, 2024*

Welcome, and thank you for your interest in MindsDB, Inc. ("MindsDB," "we," or "us") and our websites at https://mindsdb.com, https://docs.mindsdb.com, https://mdb.ai, and https://docs.mdb.ai and related website (collectively, the "Sites"). These Terms of Use are a legally binding contract between you and MindsDB regarding your use of the Sites. Please read the following terms carefully before using the Sites.

These terms do not apply to your access to or use of MindDB's products, services, or other offerings ("MindsDB Offerings"). MindsDB Offerings are subject to a separate agreements between you and MindsDB concerning such MindsDB Offerings.

By using the Sites, you acknowledge that you have read, understood, and agree to be bound by the following terms and conditions, including the MindsDB Privacy Policy at https://mindsdb.com/privacy-policy (together, the "Terms"). If you are not eligible, or you do not agree to the Terms, then you do not have our permission to use the Sites.

## Eligibility

You must be at least 18 years of age to use the Sites. By agreeing to these Terms, you represent and warrant to us that: (a) you are at least 18 years of age; and (b) your use of the Sites is in compliance with any and all applicable laws and regulations. If you are an entity, organization, or company, the individual accepting these Terms on your behalf represents and warrants that they have authority to bind you to these Terms and you agree to be bound by these Terms.

## Changes to the Terms

We may periodically make changes to these Terms. When we do, we will update the "Last Updated" date above. It is your responsibility to review the most recent version of these Terms and remain informed of any changes. You agree that your continued use of a Site after the effective date of any changes will constitute your acceptance of the changed Terms for your continued use. Disputes arising under these Terms will be resolved in accordance with the version of these Terms that was in effect at the time the dispute arose.

## Changes to the Sites

We reserve the right to modify or discontinue, temporarily or permanently, all or a part of the Sites without notice. We will not be liable to you or to any third party for any modification, suspension, or discontinuance of the Sites.

## Limited License

Subject to these Terms, MindsDB grants you a limited, revocable license to access and use the Sites to learn more about our products and services. No other use of the Sites is authorized.

## Restrictions

You must comply with all applicable laws when using the Sites. Except as may be expressly permitted by applicable law or expressly permitted by us in writing, you will not, and will not permit anyone else to: (a) store, copy, modify, distribute, or resell any information or material available on the Sites ("Site Content") or compile or collect any Site Content as part of a database or other work; (b) use any automated tool (e.g., robots, spiders) to use the Sites or store, copy, modify, distribute, or resell any Site Content; (c) rent, lease, or sublicense your access to the Sites; (d) use the Sites or Site Content for any purpose except for your own personal use; (e) circumvent or disable any digital rights management, usage rules, or other security features of the Sites; (f) reproduce, modify, translate, enhance, decompile, disassemble, reverse engineer, or create derivative works of the Sites; (g) use the Sites in a manner that threatens the integrity, performance, or availability of the Sites; or (h) remove, alter, or obscure any proprietary notices (including copyright notices) on any portion of the Sites or Site Content.

## Ownership

The Sites are owned and operated by MindsDB Inc. We or our licensors retain all right, title, and interest in and to the Sites and Site Content and any trademarks, logos, or service marks displayed on the Sites or in Site Content ("Marks"). The Sites, Site Content, and Marks are protected by applicable intellectual property laws and international treaties. Except as expressly authorized by MindsDB, you may not make use of the Sites, Site Content, and Marks.

## Privacy Policy

Please read the MindsDB Privacy Policy carefully for information relating to our collection, use, storage, disclosure of your personal information. The Privacy Policy is incorporated by this reference into, and made a part of, these Terms.

## Links and Third-Party Content

The Sites may contain links to third party products, services, and websites. We exercise no control over the third-party products, services, and websites and we are not responsible for their performance, do not endorse them, and are not responsible or liable for any content, advertising, or other materials available through the third-party products, services, and websites. We are not responsible or liable, directly or indirectly, for any damage or loss caused to you by your use of or reliance on any goods or services available through the third-party products, services, and websites.

Additionally, if you follow a link or otherwise navigate away from the Sites, please be aware that these Terms will no longer govern. You should review the applicable terms and policies, including privacy and data gathering practices, of any third-party websites to which you navigate to from the Sites.

## Feedback

MindsDB may provide you with a mechanism to provide feedback, suggestions, and ideas about the Sites or us ("Feedback"). You agree that we may, in our sole discretion, use the Feedback you provide in any way, including in future modifications to the Sites, our products, or services. You hereby grant us an unrestricted, perpetual, worldwide, fully transferable, irrevocable, royalty-free right to exploit the Feedback in any manner for any purpose.

## Disclaimer of Warranties

YOUR USE OF THE SITES AND SITE CONTENT IS AT YOUR SOLE RISK. THE SITES AND SITE CONTENT ARE PROVIDED ON AN "AS IS" AND "AS AVAILABLE" BASIS. MINDSDB EXPRESSLY DISCLAIMS ALL WARRANTIES OF ANY KIND, WHETHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. WE DO NOT GUARANTEE THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SITES OR SITE CONTENT, AND YOU RELY ON THE SITES AND SITE CONTENT AT YOUR OWN RISK. ANY MATERIAL OBTAINED THROUGH THE SITES IS DONE AT YOUR OWN DISCRETION AND RISK AND YOU WILL BE SOLELY RESPONSIBLE FOR ANY DAMAGE TO YOUR COMPUTER OR LOSS OF DATA THAT RESULTS FROM THE DOWNLOAD OF ANY MATERIAL THROUGH THE SITES. NO ADVICE OR INFORMATION, WHETHER ORAL OR WRITTEN, OBTAINED BY YOU FROM MINDSDB OR THROUGH OR FROM THE SITES WILL CREATE ANY WARRANTY NOT EXPRESSLY STATED IN THIS AGREEMENT. HOWEVER, MINDSDB DOES NOT DISCLAIM ANY WARRANTY OR OTHER RIGHT THAT MINDSDB IS PROHIBITED FROM DISCLAIMING UNDER APPLICABLE LAW.

## Limitation of Liability

MINDSDB WILL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR EXEMPLARY DAMAGES, INCLUDING BUT NOT LIMITED TO, DAMAGES FOR LOSS OF PROFITS, GOODWILL, USE, DATA OR OTHER INTANGIBLE LOSSES (EVEN IF MINDSDB HAS BEEN ADVISED OF THE POSSIBILITY OF THESE DAMAGES), RESULTING FROM YOUR USE OF THE SITES AND SITE CONTENT. UNDER NO CIRCUMSTANCES WILL MINDSDB'S TOTAL LIABILITY OF ALL KINDS ARISING OUT OF OR RELATED TO YOUR USE OF THE SITES OR SITE CONTENT (INCLUDING BUT NOT LIMITED TO WARRANTY CLAIMS), REGARDLESS OF THE FORUM AND REGARDLESS OF WHETHER ANY ACTION OR CLAIM IS BASED ON CONTRACT, TORT, OR OTHERWISE, EXCEED $50. BECAUSE SOME STATES DO NOT ALLOW THE EXCLUSION OR LIMITATION OF LIABILITY FOR CONSEQUENTIAL OR INCIDENTAL DAMAGES, THE ABOVE LIMITATION MAY NOT APPLY TO YOU.

EACH PROVISION OF THESE TERMS THAT PROVIDES FOR A LIMITATION OF LIABILITY, DISCLAIMER OF WARRANTIES, OR EXCLUSION OF DAMAGES IS INTENDED TO AND DOES ALLOCATE THE RISKS BETWEEN THE PARTIES UNDER THESE TERMS. THIS ALLOCATION IS AN ESSENTIAL ELEMENT OF THE BASIS OF THE BARGAIN BETWEEN THE PARTIES. EACH OF THESE PROVISIONS IS SEVERABLE AND INDEPENDENT OF ALL OTHER PROVISIONS OF THESE TERMS. THE LIMITATIONS IN THIS SECTION WILL APPLY EVEN IF ANY LIMITED REMEDY FAILS OF ITS ESSENTIAL PURPOSE.

## Indemnity

You will indemnify and hold MindsDB, and affiliates, officers, agents, and employees, harmless from any costs, damages, expenses, and liability caused by your use of the Sites or Site Content, your violation of these Terms, or your violation of any rights of a third party through use of the Sites or Site Content. We reserve the right, at our own expense, to assume the exclusive defense and control of any matter otherwise subject to indemnification by you (without limiting your indemnification obligations with respect to that matter), and in that case, you agree to cooperate with our defense of those claims.

## General Terms

These Terms, together with the Privacy Policy and any other agreements expressly incorporated by reference into these Terms, are the entire and exclusive understanding and agreement between you and MindsDB regarding your use of the Sites. You may not assign or transfer these Terms or your rights under these Terms, in whole or in part, by operation of law or otherwise, without our prior written consent. We may assign these Terms at any time without notice or consent. The failure to require performance of any provision will not affect our right to require performance at any other time after that, nor will a waiver by us of any breach or default of these Terms, or any provision of these Terms, be a waiver of any subsequent breach or default or a waiver of the provision itself. Use of section headers in these Terms is for convenience only and will not have any impact on the interpretation of any provision. Throughout these Terms the use of the word "including" means "including but not limited to". If any part of these Terms is held to be invalid or unenforceable, the unenforceable part will be given effect to the greatest extent possible, and the remaining parts will remain in full force and effect.

## Legal Notices

These Terms are governed by the laws of the state of California without regard to conflict of law principles. The exclusive jurisdiction and venue for any claims arising out of or related to these Terms or your use of the Sites will lie in the state and federal courts located in San Francisco, California, and you irrevocably agree to submit to the jurisdiction of such courts. The failure of MindsDB to enforce any right or provision in these Terms will not constitute a waiver of such right or provision unless acknowledged and agreed to by MindsDB in writing. In the event that a court of competent jurisdiction finds any provision of these Terms to be illegal, invalid or unenforceable, the remaining provisions will remain in full force and effect.

## Contacting MindsDB

If you have any questions or concerns about the Sites or these Terms, you may contact us by email at hello@mindsdb.com, or write to us at:

MindsDB
3277 S White Rd PMB 10166,
San Jose, CA 95148
USA
"""


PRIVACY_POLICY_MD = """\
# MindsDB Privacy Policy

*Effective date: September 4, 2025 — Last updated: September 4, 2025*

This Privacy Notice explains how MindsDB ("we," "us," or "our") collects, uses, discloses, and protects personal information when you visit our websites, interact with our marketing, register for events, contact us, or otherwise communicate with us in a business context. It also describes your privacy rights and how to exercise them.

**Who we are.** MindsDB, legal address: 3277 S White Rd PMB 10166, San Jose, CA 95148, USA.

**Data Protection Officer (DPO).** Adam Carrigan – hello@mindsdb.com.

If you are a user of our products under a separate agreement, or an employee/candidate, this website notice may be supplemented by product- or HR-specific notices.

## 1) Scope and audience

This notice applies to personal information about visitors, prospective customers and partners, vendors, and other contacts who interact with our public websites and marketing (collectively, Business Contacts). It does not cover information we process solely on behalf of our customers as a processor/service provider — those activities are governed by a separate data processing agreement.

## 2) Information we collect

We collect personal information in the categories below (examples are illustrative):

- **Identifiers & contact details** – name, employer, job title/role, business email, phone, country/region, and any information you provide in forms or communications.
- **Professional information** – company, team, interests, purchase intent, and CRM notes.
- **Internet / device & usage data** – IP address, device and browser type, operating system, pages viewed, links clicked, referring URLs, time spent, language, and approximate geolocation (derived from IP).
- **Cookies & similar technologies** – identifiers set by us and our partners for analytics, site performance, fraud/security, and (if enabled) advertising/retargeting.
- **Event & webinar data** – registration information, attendance, and feedback.
- **Support interactions** – content of messages, call/chat metadata.
- **Sensitive personal information** – we do not intentionally collect sensitive personal information via our marketing websites. If you choose to provide such information in free-text fields, we will process it only as necessary to handle your request.

**Sources of information.** We collect information directly from you (forms, emails, support), automatically via your device (cookies, logs), and from third parties (co-marketing partners, event organizers, lead-generation providers, and publicly available sources such as LinkedIn when allowed by law).

## 3) How we use personal information (purposes)

We use personal information to:

- Provide, operate, and improve our websites and services.
- Communicate with you, respond to inquiries, send administrative messages, and provide requested materials (e.g., whitepapers, demos).
- Market and advertise our services, including measuring campaign effectiveness and building audiences (see Cookies & Tracking).
- Plan and host events and manage registrations.
- Maintain security and prevent fraud, abuse, or misuse.
- Comply with legal obligations and enforce our agreements.

Where required by law, we will obtain consent before using cookies for non-essential purposes or sending you certain marketing communications.

## 4) Cookies & tracking controls

We and our partners use cookies, SDKs, and similar technologies.

In the Regions requiring prior consent (for example, the EU/UK/EEA/Switzerland/Canada), we seek opt-in consent for any non-essential cookies (e.g., analytics/advertising). These cookies are not set until you accept.

In the United States, you may opt out of interest-based advertising and the disclosure of your data for targeted advertising via the Your Privacy Choices link and by enabling recognized opt-out preference signals (see below).

You can update your cookie preferences at any time via the cookie banner, the Cookie Preferences control in our footer, or your browser settings. Blocking cookies may affect some site features.

For a current, always-up-to-date list of cookies and partners we use, see our Cookie Policy (Cookie Declaration), accessible via the cookie banner and the Cookie Preferences control in our footer.

## 5) When and with whom we disclose information

We disclose personal information as described below (we do not disclose lists of individual names unless required by law):

- **Service providers/contractors** who process data for us under written contracts (e.g., hosting, analytics, email delivery, CRM, event management, customer support, security). They may access personal information only to perform services for us and must not use it for their own purposes.
- **Advertising and analytics partners** to measure campaigns, improve our sites, and (if enabled) show you relevant ads on other sites. See Your choices below for how to opt out.
- **Business partners** in co-marketing or joint events when you register for those programs.
- **Professional advisors** (lawyers, accountants), authorities, and others when necessary for compliance, safety, and legal reasons.
- **Corporate transactions** – if we engage in a merger, acquisition, financing, or sale of assets, your information may be transferred as part of that transaction, subject to confidentiality.

We do not knowingly allow our service providers to collect, use, or disclose personal information except to provide the services we request or as permitted by law and contract.

## 6) "Selling" or "sharing" personal information / targeted advertising

We do not "sell" personal information for money. We may "share" identifiers and internet/activity data with advertising partners for cross-context behavioral advertising (targeted ads). You may opt out at any time via the Your Privacy Choices link and by using recognized opt-out preference signals (Global Privacy Control).

We honor Global Privacy Control (GPC) and other recognized regional opt-out signals where required. See Your rights & choices.

## 7) Your rights & choices

Your rights depend on where you live. We will honor requests as required by applicable law and, where reasonable, in other regions as well.

### California & other U.S. state privacy laws

If you live in CA, CO, CT, UT, VA, OR (and other states with comprehensive privacy laws), you may have the right to:

- **Know/Access** the categories and specific pieces of personal information we collected about you.
- **Delete** personal information we collected from you.
- **Correct** inaccurate personal information.
- **Opt out** of the sale or sharing of personal information and of targeted advertising.
- **Limit** the use and disclosure of sensitive personal information (if we collect it).
- **Appeal** our decision if we decline to act on your request (where applicable).

**How to exercise your rights.** Submit a request at hello@mindsdb.com (subject line: Privacy Request). For opt-outs of targeted advertising, use the Your Privacy Choices link on our site or enable GPC in your browser. We will confirm receipt within 10 business days and respond within 45 days (or as allowed by law). We will take steps to verify your identity (and authority of any authorized agent).

We will not discriminate against you for exercising your rights.

**Do Not Track.** Our websites do respond to browser Do-Not-Track (DNT) signals by disabling non-essential tracking used for targeted advertising (and limiting analytics as configured). We also honor Global Privacy Control (GPC).

### EU/EEA, UK, and Switzerland (GDPR/UK GDPR)

If you are in the EU/EEA, UK, or Switzerland, you have the rights to access, rectify, erase, restrict, object (including to direct marketing), and data portability, and the right to withdraw consent at any time (without affecting the lawfulness of processing before withdrawal). You also have the right to lodge a complaint with your local supervisory authority.

**Legal bases.** Depending on the context, we rely on: contract (to provide requested materials), legitimate interests (to operate and secure our sites, to conduct B2B marketing compatible with your expectations, to improve our services), consent (for non-essential cookies/marketing where required), and legal obligation.

**DPO contact.** Adam Carrigan – hello@mindsdb.com.

### Brazil (LGPD), Canada (PIPEDA/Provincial), Japan (APPI), South Korea (PIPA), and other regions

Residents of these jurisdictions may have similar rights to access, correct, delete, or object/opt-out of certain processing, and to lodge complaints with local authorities. You can exercise rights by contacting hello@mindsdb.com.

## 8) International data transfers

We are headquartered in the United States and may transfer your personal information to countries that may not provide the same level of protection as your home country. Where required, we use legally approved safeguards (for example, Standard Contractual Clauses (SCCs) or country-specific equivalents).

## 9) Data retention

We keep personal information for as long as necessary to achieve the purposes described above, including to comply with legal, accounting, or reporting obligations, resolve disputes, and enforce agreements. We also consider the amount, nature, and sensitivity of the data and the risk of harm from unauthorized use or disclosure.

Typical retention periods:

- Website analytics: 13–24 months (per cookie/tool settings).
- CRM & marketing records for Business Contacts: 3 years after last meaningful interaction.
- Event registration: 3 years after the event.
- Support communications: 5 years after resolution.

## 10) Security

We use administrative, technical, and physical measures designed to protect personal information. No method of transmission or storage is 100% secure; if we learn of a security incident that affects your information, we will notify you as required by law.

## 11) Children's privacy

Our websites are intended for adults and are not directed to children. We do not knowingly collect personal information from anyone under 16. If you believe a child has provided personal information to us, contact hello@mindsdb.com and we will take appropriate steps.

## 12) Changes to this notice

We may update this notice from time to time. When we do, we will change the "Last updated" date at the top and, when required, provide additional notice.

## 13) How to contact us

MindsDB
Legal address: 3277 S White Rd PMB 10166, San Jose, CA 95148, USA
Data Protection Officer: Adam Carrigan – hello@mindsdb.com
For privacy requests, email hello@mindsdb.com and include Privacy Request in the subject line.
"""
