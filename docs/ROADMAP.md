# Roadmap

## Done (v0.1)

- [x] Scrape National Assembly and Senate bill listings (all pages)
- [x] Extract text from PDFs; detect scans
- [x] Structured English + Kiswahili summaries with clause references
- [x] Draft → human review → publish workflow
- [x] Read-only public API

## Next

1. **Public participation deadlines.** Parse Kenya Gazette notices and committee adverts for submission deadlines,
   and attach them to bills.
2. **WhatsApp and SMS alerts.** Weekly digest and deadline reminders via Africa's Talking (SMS) and the WhatsApp
   Cloud API. Subscribers pick topics (tax, health, land, education) and a language.
3. **Bill stages.** Track first reading → committee → second reading → assent from the Order Papers and the "Bill
   Tracker" links on parliament.go.ke.
4. **Feed Public-Participation-Platform.** Export a published summary as a draft consultation (Kiswahili summary
   and full document are already required there), so county and national consultations start with a plain-language
   explainer.
5. **County assemblies.** Start with Kisumu (the Public-Participation-Platform pilot county), then Nairobi and
   Mombasa.
6. **OCR for very large scans** (over 32 MB) so they can be summarised too.
7. **Web frontend.** Simple, fast pages that work on low-end phones, with a Kiswahili-first toggle.

## Open questions

- Who reviews summaries? Partnering with a law school clinic or a civic organisation such as Mzalendo would
  add credibility.
- Hosting and Claude API costs: estimate per bill, then look for a civic tech grant.
