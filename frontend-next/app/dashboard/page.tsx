import { KpiCards } from "@/components/kpi-cards";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DailyActiveUsersChart,
  ModelUsageBarChart,
  ModelUsagePieChart,
  TokenConsumptionChart,
  UserAdoptionChart
} from "@/components/charts/overview-charts";
import { apiGet } from "@/lib/api";
import {
  buildChatUserMap,
  buildDailyActiveUsersSeries,
  buildModelUsageBreakdown,
  buildTokenConsumptionSeries,
  buildUserAdoptionSeries,
  calculateEngagementMetrics,
  computeDateSummary,
  normaliseChats,
  normaliseMessages,
  normaliseModels,
  normaliseUsers
} from "@/lib/overview";

export default async function DashboardOverviewPage() {
  async function fetchOverviewData() {
    try {
      const [rawChats, rawMessages, rawUsers, rawModels] = await Promise.all([
        apiGet<unknown>("api/v1/chats"),
        apiGet<unknown>("api/v1/messages"),
        apiGet<unknown>("api/v1/users"),
        apiGet<unknown>("api/v1/models")
      ]);
      return { rawChats, rawMessages, rawUsers, rawModels };
    } catch {
      return null;
    }
  }

  const data = await fetchOverviewData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load overview</CardTitle>
          <CardDescription>We could not reach the backend API. Please verify the service is running.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const usersMap = normaliseUsers(data.rawUsers);
  const modelsMap = normaliseModels(data.rawModels);
  const chats = normaliseChats(data.rawChats);
  const messages = normaliseMessages(data.rawMessages, modelsMap);

  if (!messages.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No data available yet</CardTitle>
          <CardDescription>Upload chat exports or sync with Open WebUI to populate the overview.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Once chats and messages are ingested, you&apos;ll see high level metrics and charts here that mirror the
            legacy overview experience.
          </p>
        </CardContent>
      </Card>
    );
  }

  const chatUserMap = buildChatUserMap(chats, usersMap);
  const metrics = calculateEngagementMetrics(chats, messages);
  const dateSummary = computeDateSummary(messages);
  const tokenSeries = buildTokenConsumptionSeries(messages);
  const modelBreakdown = buildModelUsageBreakdown(messages);
  const modelPie = modelBreakdown.map(({ model, count }) => ({ name: model, value: count }));
  const adoptionSeries = buildUserAdoptionSeries(messages, chatUserMap, dateSummary.dateMin, dateSummary.dateMax);
  const dailyActiveUsersSeries = buildDailyActiveUsersSeries(messages, chats, chatUserMap, dateSummary.dateMin, dateSummary.dateMax);

  const avgFormat: Intl.NumberFormatOptions = { minimumFractionDigits: 1, maximumFractionDigits: 1 };

  const primaryKpis = [
    {
      title: "Days",
      value: dateSummary.totalDays.toLocaleString(),
      description: "Distinct days with message activity."
    },
    {
      title: "Total Chats",
      value: metrics?.totalChats.toLocaleString() ?? "0",
      description: "Chat threads ingested."
    },
    {
      title: "Unique Users",
      value: metrics?.uniqueUsers.toLocaleString() ?? "0",
      description: "Users contributing to chats."
    },
    {
      title: "Daily Active Users",
      value: metrics ? metrics.avgDailyActiveUsers.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Average users active per day over the past 30 days."
    }
  ];

  const featureUsageKpis = [
    {
      title: "Files Uploaded",
      value: metrics?.filesUploaded.toLocaleString() ?? "0",
      description: "Attachments captured across chats."
    },
    {
      title: "Web Searches",
      value: metrics?.webSearchesUsed.toLocaleString() ?? "0",
      description: "Chats that used web search functionality."
    },
    {
      title: "Knowledge Base",
      value: metrics?.knowledgeBaseUsed.toLocaleString() ?? "0",
      description: "Chats that used knowledge base retrieval."
    }
  ];

  const secondaryKpis = [
    {
      title: "Avg Msgs/Chat",
      value: metrics ? metrics.avgMessagesPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Conversation length per chat."
    },
    {
      title: "Avg Input Tokens/Chat",
      value: metrics ? metrics.avgInputTokensPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Average user input tokens per chat."
    },
    {
      title: "Avg Output Tokens/Chat",
      value: metrics ? metrics.avgOutputTokensPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Average assistant response tokens per chat."
    },
    {
      title: "Total Tokens",
      value: metrics?.totalTokens.toLocaleString() ?? "0",
      description: "Total input and output tokens exchanged."
    }
  ];

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">Overview</h1>
            <p className="text-sm text-muted-foreground">
              Date Range: {dateSummary.dateMinLabel} – {dateSummary.dateMaxLabel}
            </p>
          </div>
        </div>
        <KpiCards items={primaryKpis} />
        <KpiCards items={secondaryKpis} />
      </section>

      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-foreground">Feature Usage Overview</h2>
          <p className="text-sm text-muted-foreground">
            Advanced features utilized across conversations.
          </p>
        </div>
        <KpiCards items={featureUsageKpis} gridClassName="grid gap-4 md:grid-cols-3" />
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Daily Token Consumption</CardTitle>
          <CardDescription>Total tokens generated over time.</CardDescription>
        </CardHeader>
        <CardContent>
          {tokenSeries.length ? (
            <TokenConsumptionChart data={tokenSeries} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Token consumption chart unavailable — insufficient timestamped data.
            </p>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Assistant Responses by Model</CardTitle>
            <CardDescription>Selection of models producing assistant replies.</CardDescription>
          </CardHeader>
          <CardContent>
            {modelBreakdown.length ? (
              <ModelUsageBarChart data={modelBreakdown} />
            ) : (
              <p className="text-sm text-muted-foreground">No assistant messages with model information available.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Usage Share</CardTitle>
            <CardDescription>Relative share of assistant responses by model.</CardDescription>
          </CardHeader>
          <CardContent>
            {modelPie.length ? (
              <ModelUsagePieChart data={modelPie} />
            ) : (
              <p className="text-sm text-muted-foreground">
                We need assistant messages with model metadata to build this chart.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Cumulative User Adoption</CardTitle>
          <CardDescription>First-message milestones for each user.</CardDescription>
        </CardHeader>
        <CardContent>
          {adoptionSeries.length ? (
            <UserAdoptionChart data={adoptionSeries} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Need user-authored messages with timestamps to plot adoption over time.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Daily Active Users</CardTitle>
          <CardDescription>Number of users who created a conversation or sent a message each day.</CardDescription>
        </CardHeader>
        <CardContent>
          {dailyActiveUsersSeries.length ? (
            <DailyActiveUsersChart data={dailyActiveUsersSeries} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Need user activity data with timestamps to plot daily active users over time.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="text-sm leading-relaxed text-muted-foreground">
          Navigate with the sidebar to explore the detailed dashboards: Time Analysis, Content Analysis, Sentiment,
          Search Chats, and Browse Chats.
        </CardContent>
      </Card>
    </div>
  );
}
