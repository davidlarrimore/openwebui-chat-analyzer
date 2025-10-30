import { redirect } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { KpiCards } from "@/components/kpi-cards";
import { ModelUsageBarChart } from "@/components/charts/overview-charts";
import {
  ModelUsageShareBarChart,
  ModelUsageTimelineChart,
  ModelAverageTokensChart,
  ModelTopicStackedBarChart
} from "@/components/charts/model-analysis-charts";
import { ApiError, apiGet } from "@/lib/api";
import { normaliseChats, normaliseMessages, normaliseModels, buildModelUsageBreakdown } from "@/lib/overview";
import {
  buildModelUsageShare,
  buildModelUsageTimeline,
  buildAverageTokensByModel,
  buildModelTopicDistribution
} from "@/lib/model-analysis";

interface RawModelPayload {
  rawChats: unknown;
  rawMessages: unknown;
  rawModels: unknown;
}

async function fetchModelData(): Promise<RawModelPayload | null> {
  try {
    const [rawChats, rawMessages, rawModels] = await Promise.all([
      apiGet<unknown>("api/v1/chats"),
      apiGet<unknown>("api/v1/messages"),
      apiGet<unknown>("api/v1/models")
    ]);
    return { rawChats, rawMessages, rawModels };
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      redirect(`/login?error=AuthRequired&callbackUrl=${encodeURIComponent("/dashboard/models")}`);
    }
    return null;
  }
}

export default async function ModelAnalysisPage() {
  const data = await fetchModelData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load model analytics</CardTitle>
          <CardDescription>We could not reach the backend API. Please verify the service is running.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const modelsMap = normaliseModels(data.rawModels);
  const chats = normaliseChats(data.rawChats);
  const messages = normaliseMessages(data.rawMessages, modelsMap);

  const modelBreakdownAsc = buildModelUsageBreakdown(messages);
  if (!modelBreakdownAsc.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No assistant responses yet</CardTitle>
          <CardDescription>
            Upload chat exports or sync with Open WebUI to populate model analytics across the dashboard.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Once assistant responses with model metadata are ingested you&apos;ll see expanded usage trends, timelines,
            and topic insights for your deployment here.
          </p>
        </CardContent>
      </Card>
    );
  }

  const modelBreakdown = [...modelBreakdownAsc].sort((a, b) => b.count - a.count);
  const totalAssistantResponses = modelBreakdown.reduce((acc, item) => acc + item.count, 0);
  const distinctModels = modelBreakdown.length;

  const usageShare = buildModelUsageShare(modelBreakdown, 12);
  const timeline = buildModelUsageTimeline(messages, modelBreakdown, { topModelLimit: 6 });
  const averageTokens = buildAverageTokensByModel(messages, modelBreakdown, { topModelLimit: 8 });
  const topicDistribution = buildModelTopicDistribution(messages, chats, modelBreakdown, {
    topModelLimit: 5,
    topTagLimit: 8
  });

  const taggedChatCount = chats.filter((chat) => Array.isArray(chat.tags) && chat.tags.length > 0).length;
  const topicCoverage = chats.length > 0 ? (taggedChatCount / chats.length) * 100 : 0;

  const kpiItems = [
    {
      title: "Assistant Responses",
      value: totalAssistantResponses.toLocaleString(),
      description: "Total assistant messages with model metadata."
    },
    {
      title: "Distinct Models",
      value: distinctModels.toLocaleString(),
      description: "Unique assistant models observed."
    },
    {
      title: "Tagged Chat Coverage",
      value: `${topicCoverage.toFixed(1)}%`,
      description: "Chats with tags available for topic analysis."
    }
  ];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Model Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Deep dive into model adoption, performance signals, and topic patterns across assistant interactions.
        </p>
      </header>

      <KpiCards items={kpiItems} />

      <div className="grid gap-6 xl:grid-cols-[2fr,1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Assistant Responses by Model</CardTitle>
            <CardDescription>Overall volume of assistant replies per model across all chats.</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelUsageBarChart data={modelBreakdown} height={400} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Usage Share</CardTitle>
            <CardDescription>Percentage share of assistant responses by model (top 12 consolidated).</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelUsageShareBarChart data={usageShare} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Model Usage Timeline</CardTitle>
          <CardDescription>Daily assistant message volume for the most active models.</CardDescription>
        </CardHeader>
        <CardContent>
          <ModelUsageTimelineChart data={timeline.data} models={timeline.models} />
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Average Output Tokens by Model</CardTitle>
            <CardDescription>Mean assistant response length by model, using approximate token counts.</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelAverageTokensChart data={averageTokens} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Topic Patterns by Model</CardTitle>
            <CardDescription>Top tagged chat themes and the models most frequently responding to them.</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelTopicStackedBarChart data={topicDistribution.data} models={topicDistribution.models} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
