import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export interface KpiItem {
  title: string;
  value: string;
  description?: string;
}

interface KpiCardsProps {
  items: KpiItem[];
}

export function KpiCards({ items }: KpiCardsProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {items.map((item) => (
        <Card key={item.title}>
          <CardHeader className="space-y-1">
            <CardTitle className="text-sm font-medium text-muted-foreground">{item.title}</CardTitle>
            <CardDescription className="text-2xl font-semibold text-foreground">{item.value}</CardDescription>
          </CardHeader>
          {item.description && (
            <CardContent>
              <p className="text-xs text-muted-foreground">{item.description}</p>
            </CardContent>
          )}
        </Card>
      ))}
    </div>
  );
}
