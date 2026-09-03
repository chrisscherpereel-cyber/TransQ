"""QTI export.

Two dialects are produced, because "QTI" in practice means two different things:

* ``export_qti12_canvas`` — QTI 1.2 in the Canvas-flavored package layout
  (imsmanifest + questestinterop + assessment_meta). This is what Canvas's
  "Import Course Content -> QTI .zip file" path handles most reliably, and it is
  what Blackboard, D2L, and Moodle's QTI importers generally accept too.
* ``export_qti21`` — standards-clean IMS QTI 2.1 items plus a manifest, for
  tools that require 2.1.

Start with the 1.2 package. Fall back to 2.1 only if your LMS rejects it.
"""

from __future__ import annotations

import html
import io
import uuid
import zipfile
from xml.sax.saxutils import escape

from ..schema import Quiz

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _ident(prefix: str = "i") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def _html_text(text: str) -> str:
    """Wrap plain text as escaped HTML for a QTI mattext node."""
    safe = html.escape(text or "", quote=False).replace("\n", "<br/>")
    return escape(f"<p>{safe}</p>")


def _xml(text: str) -> str:
    return escape(text or "", {'"': "&quot;", "'": "&apos;"})


# --------------------------------------------------------------------------- #
# QTI 1.2 (Canvas)
# --------------------------------------------------------------------------- #


def _qti12_item(q, number: int) -> str:
    item_id = _ident("q")
    resp_ids = [_ident("a") for _ in q.options]
    correct_id = resp_ids[q.correct_index]

    labels = "\n".join(
        f"""            <response_label ident="{rid}">
              <material><mattext texttype="text/html">{_html_text(opt)}</mattext></material>
            </response_label>"""
        for rid, opt in zip(resp_ids, q.options)
    )

    feedback_blocks = ""
    conditions = ""
    if q.rationale:
        feedback_blocks += f"""
    <itemfeedback ident="correct_fb">
      <flow_mat><material><mattext texttype="text/html">{_html_text(q.rationale)}</mattext></material></flow_mat>
    </itemfeedback>"""

    for rid, rat in zip(resp_ids, (q.distractor_rationales or [""] * len(q.options))):
        if rid == correct_id or not rat:
            continue
        fb_id = f"{rid}_fb"
        feedback_blocks += f"""
    <itemfeedback ident="{fb_id}">
      <flow_mat><material><mattext texttype="text/html">{_html_text(rat)}</mattext></material></flow_mat>
    </itemfeedback>"""
        conditions += f"""
        <respcondition continue="Yes">
          <conditionvar><varequal respident="response1">{rid}</varequal></conditionvar>
          <displayfeedback feedbacktype="Response" linkrefid="{fb_id}"/>
        </respcondition>"""

    title = _xml(q.topic or f"Question {number}")

    return f"""      <item ident="{item_id}" title="{title}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>question_type</fieldlabel>
              <fieldentry>multiple_choice_question</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>points_possible</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>assessment_question_identifierref</fieldlabel>
              <fieldentry>{_ident("aq")}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">{_html_text(q.stem)}</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
{labels}
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar maxvalue="100" minvalue="0" varname="SCORE" vartype="Decimal"/>
          </outcomes>{conditions}
          <respcondition continue="No">
            <conditionvar><varequal respident="response1">{correct_id}</varequal></conditionvar>
            <setvar action="Set" varname="SCORE">100</setvar>
            {'<displayfeedback feedbacktype="Response" linkrefid="correct_fb"/>' if q.rationale else ''}
          </respcondition>
        </resprocessing>{feedback_blocks}
      </item>"""


def export_qti12_canvas(quiz: Quiz) -> bytes:
    """Build a Canvas-importable QTI 1.2 .zip."""
    questions = quiz.included
    quiz_id = _ident("g")
    meta_id = _ident("m")
    items = "\n".join(_qti12_item(q, i) for i, q in enumerate(questions, start=1))

    assessment = f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:schemaLocation="http://www.imsglobal.org/xsd/ims_qtiasiv1p2 http://www.imsglobal.org/xsd/ims_qtiasiv1p2p1.xsd">
  <assessment ident="{quiz_id}" title="{_xml(quiz.meta.title)}">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_maxattempts</fieldlabel>
        <fieldentry>1</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="root_section">
{items}
    </section>
  </assessment>
</questestinterop>
"""

    description = quiz.meta.description or (
        f"Generated from {quiz.meta.source_filename}" if quiz.meta.source_filename else ""
    )
    assessment_meta = f"""<?xml version="1.0" encoding="UTF-8"?>
<quiz identifier="{quiz_id}"
      xmlns="http://canvas.instructure.com/xsd/cccv1p0"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0 https://canvas.instructure.com/xsd/cccv1p0.xsd">
  <title>{_xml(quiz.meta.title)}</title>
  <description>{_xml(description)}</description>
  <shuffle_answers>false</shuffle_answers>
  <scoring_policy>keep_highest</scoring_policy>
  <hide_results></hide_results>
  <quiz_type>assignment</quiz_type>
  <points_possible>{quiz.total_points}</points_possible>
  <require_lockdown_browser>false</require_lockdown_browser>
  <require_lockdown_browser_for_results>false</require_lockdown_browser_for_results>
  <allowed_attempts>1</allowed_attempts>
  <one_question_at_a_time>false</one_question_at_a_time>
  <cant_go_back>false</cant_go_back>
  <available>false</available>
  <one_time_results>false</one_time_results>
  <show_correct_answers_last_attempt>false</show_correct_answers_last_attempt>
  <only_visible_to_overrides>false</only_visible_to_overrides>
  <module_locked>false</module_locked>
</quiz>
"""

    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="{_ident("man")}"
          xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1 http://www.imsglobal.org/profile/cc/ccv1p1/ccv1p1_imscp_v1p2_v1p0.xsd">
  <metadata>
    <schema>IMS Content</schema>
    <schemaversion>1.1.3</schemaversion>
  </metadata>
  <organizations/>
  <resources>
    <resource identifier="{quiz_id}" type="imsqti_xmlv1p2" href="{quiz_id}/{quiz_id}.xml">
      <file href="{quiz_id}/{quiz_id}.xml"/>
      <dependency identifierref="{meta_id}"/>
    </resource>
    <resource identifier="{meta_id}"
              type="associatedcontent/imscc_xmlv1p1/learning-application-resource"
              href="{quiz_id}/assessment_meta.xml">
      <file href="{quiz_id}/assessment_meta.xml"/>
    </resource>
  </resources>
</manifest>
"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr(f"{quiz_id}/{quiz_id}.xml", assessment)
        zf.writestr(f"{quiz_id}/assessment_meta.xml", assessment_meta)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# QTI 2.1
# --------------------------------------------------------------------------- #


def _qti21_item(q, number: int) -> str:
    choice_ids = [f"Choice{chr(65 + i)}" for i in range(len(q.options))]
    correct = choice_ids[q.correct_index]
    choices = "\n".join(
        f'      <simpleChoice identifier="{cid}">{escape(opt)}</simpleChoice>'
        for cid, opt in zip(choice_ids, q.options)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<assessmentItem xmlns="http://www.imsglobal.org/xsd/imsqti_v2p1"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.imsglobal.org/xsd/imsqti_v2p1 http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1.xsd"
                identifier="item{number}"
                title="{_xml(q.topic or f'Question {number}')}"
                adaptive="false"
                timeDependent="false">
  <responseDeclaration identifier="RESPONSE" cardinality="single" baseType="identifier">
    <correctResponse>
      <value>{correct}</value>
    </correctResponse>
  </responseDeclaration>
  <outcomeDeclaration identifier="SCORE" cardinality="single" baseType="float">
    <defaultValue><value>0</value></defaultValue>
  </outcomeDeclaration>
  <itemBody>
    <choiceInteraction responseIdentifier="RESPONSE" shuffle="false" maxChoices="1">
      <prompt>{escape(q.stem)}</prompt>
{choices}
    </choiceInteraction>
  </itemBody>
  <responseProcessing template="http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct"/>
</assessmentItem>
"""


def export_qti21(quiz: Quiz) -> bytes:
    """Build a standards-clean IMS QTI 2.1 item package."""
    questions = quiz.included
    files: list[tuple[str, str]] = []
    for i, q in enumerate(questions, start=1):
        files.append((f"items/item{i}.xml", _qti21_item(q, i)))

    resources = "\n".join(
        f"""    <resource identifier="item{i}" type="imsqti_item_xmlv2p1" href="items/item{i}.xml">
      <file href="items/item{i}.xml"/>
    </resource>"""
        for i in range(1, len(questions) + 1)
    )

    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="{_ident("man")}"
          xmlns="http://www.imsglobal.org/xsd/imscp_v1p1"
          xmlns:imsmd="http://www.imsglobal.org/xsd/imsmd_v1p2"
          xmlns:imsqti="http://www.imsglobal.org/xsd/imsqti_v2p1"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://www.imsglobal.org/xsd/imscp_v1p1 http://www.imsglobal.org/xsd/imscp_v1p2.xsd">
  <metadata>
    <schema>IMS Content</schema>
    <schemaversion>1.2</schemaversion>
  </metadata>
  <organizations/>
  <resources>
{resources}
  </resources>
</manifest>
"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        for name, content in files:
            zf.writestr(name, content)
    return buffer.getvalue()
