import { KpiCards } from "@/components/kpi-cards";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ModelUsageBarChart,
  ModelUsagePieChart,
  TokenConsumptionChart,
  UserAdoptionChart
} from "@/components/charts/overview-charts";
import { apiGet } from "@/lib/api";
import {
  buildChatUserMap,
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
      title: "User Files Uploaded",
      value: metrics?.filesUploaded.toLocaleString() ?? "0",
      description: "Attachments captured across chats."
    }
  ];

  const secondaryKpis = [
    {
      title: "Avg Msgs/Chat",
      value: metrics ? metrics.avgMessagesPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Conversation length per chat."
    },
    {
      title: "Avg Input Characters/Chat",
      value: metrics ? metrics.avgInputTokensPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Average user input characters per chat."
    },
    {
      title: "Avg Output Characters/Chat",
      value: metrics ? metrics.avgOutputTokensPerChat.toLocaleString(undefined, avgFormat) : "0.0",
      description: "Average assistant response characters per chat."
    },
    {
      title: "Total Characters",
      value: metrics?.totalTokens.toLocaleString() ?? "0",
      description: "Total input and output characters exchanged."
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

      <Card>
        <CardHeader>
          <CardTitle>Daily Character Consumption</CardTitle>
          <CardDescription>Total characters generated over time.</CardDescription>
        </CardHeader>
        <CardContent>
          {tokenSeries.length ? (
            <TokenConsumptionChart data={tokenSeries} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Character consumption chart unavailable — insufficient timestamped data.
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
        <CardContent className="text-sm leading-relaxed text-muted-foreground">
          Navigate with the sidebar to explore the detailed dashboards: Time Analysis, Content Analysis, Sentiment,
          Search Chats, and Browse Chats.
        </CardContent>
      </Card>
    </div>
  );
}
