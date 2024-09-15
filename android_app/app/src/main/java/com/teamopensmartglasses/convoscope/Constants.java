package com.teamopensmartglasses.convoscope;

public class Constants {
    public static String appName = "Convoscope";
    public static String glassesCardTitle = "";
    public static String displayRequestsKey = "display_requests";
    public static String proactiveAgentResultsKey = "results_proactive_agent_insights";
    public static String explicitAgentQueriesKey = "explicit_insight_queries";
    public static String explicitAgentResultsKey = "explicit_insight_results";
    public static String wakeWordTimeKey = "wake_word_time";
    public static String entityDefinitionsKey = "entity_definitions";
    public static String languageLearningKey = "language_learning_results";
    public static String llContextConvoKey = "ll_context_convo_results";
    public static String llWordSuggestUpgradeKey = "ll_word_suggest_upgrade_results";
    public static String shouldUpdateSettingsKey = "should_update_settings";
    public static String adhdStmbAgentKey = "adhd_stmb_agent_results";

    //endpoints
    public static final String LLM_QUERY_ENDPOINT = "/chat";
    public static final String DIARIZE_QUERY_ENDPOINT = "/chat_diarization";
    public static final String GEOLOCATION_STREAM_ENDPOINT = "/gps_location";
    public static final String BUTTON_EVENT_ENDPOINT = "/button_event";
    public static final String UI_POLL_ENDPOINT = "/ui_poll";
    public static final String SET_USER_SETTINGS_ENDPOINT = "/set_user_settings";
    public static final String GET_USER_SETTINGS_ENDPOINT = "/get_user_settings";
}
