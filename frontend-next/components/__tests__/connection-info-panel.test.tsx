/**
 * Tests for ConnectionInfoPanel component
 *
 * These tests verify:
 * - Fields are disabled by default and require "Edit Data Source" to enable
 * - Save/Cancel buttons appear only in edit mode
 * - Mode selector and Last Sync render based on sync status
 * - Test Connection shows success/failure states
 * - Load Data triggers correct endpoint
 * - Staleness pill shows correct state
 * - Scheduler drawer functionality
 */

import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { ConnectionInfoPanel } from "../connection-info-panel";
import * as api from "@/lib/api";
import * as health from "@/lib/health";

// Mock the API module
jest.mock("@/lib/api");
const mockedApi = api as jest.Mocked<typeof api>;
jest.mock("@/lib/health");
const mockedHealth = health as jest.Mocked<typeof health>;

// Mock toast
jest.mock("@/components/ui/use-toast", () => ({
  toast: jest.fn(),
}));

jest.mock("@/components/summarizer-progress-provider", () => ({
  useSummarizerProgress: () => ({ updateStatus: jest.fn() }),
  SummarizerProgressProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const flushPromises = () => new Promise(resolve => setTimeout(resolve, 0));

async function renderConnectionInfoPanel(props?: React.ComponentProps<typeof ConnectionInfoPanel>) {
  const utils = render(<ConnectionInfoPanel {...props} />);
  for (let i = 0; i < 3; i += 1) {
    await act(async () => {
      await flushPromises();
    });
  }
  return utils;
}

const defaultInitialSettings = {
  host: "http://test.com",
  api_key: "test-key",
  database_host: "http://test.com",
  database_api_key: "test-key",
  host_source: "database" as const,
  api_key_source: "database" as const,
};


describe("ConnectionInfoPanel", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Default mock implementations
    mockedApi.getSyncStatus.mockResolvedValue({
      last_sync_at: "2024-01-01T12:00:00Z",
      last_watermark: "2024-01-01T12:00:00Z",
      has_data: true,
      recommended_mode: "incremental",
      is_stale: false,
      staleness_threshold_hours: 6,
      local_counts: {
        chats: 100,
        messages: 500,
        users: 10,
        models: 5,
      },
    });

    mockedApi.getSyncScheduler.mockResolvedValue({
      enabled: false,
      interval_minutes: 60,
      last_run_at: null,
      next_run_at: null,
    });

    mockedApi.getAnonymizationSettings.mockResolvedValue({ enabled: true, source: "database" });
    mockedApi.updateAnonymizationSettings.mockResolvedValue({ enabled: true, source: "database" });
    mockedApi.getOpenWebUISettings.mockResolvedValue(defaultInitialSettings);
    mockedApi.updateOpenWebUISettings.mockResolvedValue(defaultInitialSettings);
    mockedHealth.fetchHealthStatus.mockResolvedValue({
      service: "openwebui",
      status: "ok",
      attempts: 1,
      elapsed_seconds: 0.1,
      meta: { host: "http://test.com", chat_count: 0 },
    });
    mockedApi.apiGet.mockImplementation(async (path: string) => {
      if (path === "api/v1/datasets/meta") {
        return { chat_count: 0, user_count: 0, model_count: 0, message_count: 0 };
      }
      return {} as never;
    });
    mockedApi.getSummaryStatus.mockResolvedValue(null);
    mockedApi.getSummarizerSettings.mockResolvedValue({
      model: "llama3.2:3b-instruct-q4_K_M",
      temperature: 0.2,
      model_source: "database",
      temperature_source: "database",
    });
    mockedApi.getAvailableOllamaModels.mockResolvedValue([]);
  });

describe("Edit Mode", () => {
  it("should have fields disabled by default", async () => {
    await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

    const hostnameInput = await screen.findByLabelText(/hostname/i);
    const apiKeyInput = screen.getByLabelText(/api key/i);

    expect(hostnameInput).toBeDisabled();
    expect(apiKeyInput).toBeDisabled();
  });

  it("should enable fields when Edit Credentials is clicked", async () => {
    await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

    const editButton = await screen.findByRole("button", { name: /edit credentials/i });
    fireEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByLabelText(/hostname/i)).not.toBeDisabled();
      expect(screen.getByLabelText(/api key/i)).not.toBeDisabled();
    });
  });

  it("should show Save Changes and Cancel buttons in edit mode", async () => {
    await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

    expect(screen.queryByRole("button", { name: /save changes/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/^cancel$/i)).not.toBeInTheDocument();

    const editButton = await screen.findByRole("button", { name: /edit credentials/i });
    fireEvent.click(editButton);

    expect(await screen.findByRole("button", { name: /save changes/i })).toBeInTheDocument();
    expect(screen.getByText(/^cancel$/i)).toBeInTheDocument();
  });

  it("should call API and exit edit mode when Save Changes is clicked", async () => {
    mockedApi.updateOpenWebUISettings.mockResolvedValue({
      host: "http://new-host.com",
      api_key: "new-key",
      database_host: "http://new-host.com",
      database_api_key: "new-key",
      host_source: "database",
      api_key_source: "database",
    });

    await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

    fireEvent.click(await screen.findByRole("button", { name: /edit credentials/i }));

    const hostnameInput = await screen.findByLabelText(/hostname/i);
    fireEvent.change(hostnameInput, { target: { value: "http://new-host.com" } });

    const saveButton = screen.getByRole("button", { name: /save changes/i });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockedApi.updateOpenWebUISettings).toHaveBeenCalledWith({
        host: "http://new-host.com",
      });
    });

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /save changes/i })).not.toBeInTheDocument();
    });
  });

  it("should revert form values when Cancel is clicked", async () => {
    await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

    fireEvent.click(await screen.findByRole("button", { name: /edit credentials/i }));

    const hostnameInput = (await screen.findByLabelText(/hostname/i)) as HTMLInputElement;
    fireEvent.change(hostnameInput, { target: { value: "http://changed.com" } });
    expect(hostnameInput.value).toBe("http://changed.com");

    fireEvent.click(screen.getByText(/^cancel$/i));

    await waitFor(() => {
      expect(hostnameInput.value).toBe("http://test.com");
    });

    expect(screen.queryByText(/^cancel$/i)).not.toBeInTheDocument();
  });
});

describe("Sync Status Display", () => {
  it("should display Last Sync timestamp", async () => {
    await renderConnectionInfoPanel();

    expect(await screen.findByText(/last sync/i)).toBeInTheDocument();
    expect(screen.getByText(/current/i)).toBeInTheDocument();
  });

  it("should display Current pill when data is fresh", async () => {
    mockedApi.getSyncStatus.mockResolvedValue({
      last_sync_at: "2024-01-01T12:00:00Z",
      last_watermark: "2024-01-01T12:00:00Z",
      has_data: true,
      recommended_mode: "incremental",
      is_stale: false,
      staleness_threshold_hours: 6,
      local_counts: { chats: 100, messages: 500, users: 10, models: 5 },
    });

    await renderConnectionInfoPanel();

    expect(await screen.findByText(/current/i)).toBeInTheDocument();
  });

  it("should display Stale pill when data is stale", async () => {
    mockedApi.getSyncStatus.mockResolvedValue({
      last_sync_at: "2024-01-01T00:00:00Z",
      last_watermark: "2024-01-01T00:00:00Z",
      has_data: true,
      recommended_mode: "incremental",
      is_stale: true,
      staleness_threshold_hours: 6,
      local_counts: { chats: 100, messages: 500, users: 10, models: 5 },
    });

    await renderConnectionInfoPanel();

    expect(await screen.findByText(/stale/i)).toBeInTheDocument();
  });

  it("should render data source configuration section", async () => {
    await renderConnectionInfoPanel();

    expect(await screen.findByText(/data source configuration/i)).toBeInTheDocument();
  });
});

describe("Anonymization Settings", () => {
  it("should prompt before disabling anonymization", async () => {
    await renderConnectionInfoPanel();

    const toggle = await screen.findByRole("switch", { name: /anonymization mode/i });
    fireEvent.click(toggle);

    expect(await screen.findByRole("heading", { name: /turn off anonymization/i })).toBeInTheDocument();
    expect(mockedApi.updateAnonymizationSettings).not.toHaveBeenCalled();

    const confirmButton = screen.getByRole("button", { name: /yes, turn off anonymization/i });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(mockedApi.updateAnonymizationSettings).toHaveBeenCalledWith({ enabled: false });
    });
  });

  it("should allow cancelling the anonymization warning", async () => {
    await renderConnectionInfoPanel();

    const toggle = await screen.findByRole("switch", { name: /anonymization mode/i });
    fireEvent.click(toggle);

    const cancelButton = await screen.findByRole("button", { name: /keep anonymization on/i });
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByText(/turn off anonymization/i)).not.toBeInTheDocument();
    });

    expect(mockedApi.updateAnonymizationSettings).not.toHaveBeenCalled();
  });

  it("should enable anonymization immediately when toggled on", async () => {
    mockedApi.getAnonymizationSettings.mockResolvedValueOnce({ enabled: false, source: "database" });
    mockedApi.updateAnonymizationSettings.mockResolvedValue({ enabled: true, source: "database" });

    await renderConnectionInfoPanel();

    const toggle = await screen.findByRole("switch", { name: /anonymization mode/i });
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(mockedApi.updateAnonymizationSettings).toHaveBeenCalledWith({ enabled: true });
    });

    expect(screen.queryByText(/turn off anonymization/i)).not.toBeInTheDocument();
  });
});

describe("Test Connection", () => {
  it("should show success toast on successful connection", async () => {
    const { toast } = require("@/components/ui/use-toast");

    mockedApi.testOpenWebUIConnection.mockResolvedValue({
      service: "openwebui",
      status: "ok",
      attempts: 1,
      elapsed_seconds: 0.5,
      meta: {
        version: "0.1.0",
        chat_count: 50,
      },
    });

    await renderConnectionInfoPanel();

    const testButton = await screen.findByText(/test connection/i);
    fireEvent.click(testButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Connection Successful",
            variant: "default",
          })
        );
      });
    });

    it("should show error toast on failed connection", async () => {
      const { toast } = require("@/components/ui/use-toast");

      mockedApi.testOpenWebUIConnection.mockResolvedValue({
        service: "openwebui",
        status: "error",
        attempts: 3,
        elapsed_seconds: 10.0,
        detail: "Connection timeout",
      });

    await renderConnectionInfoPanel();

    const testButton = await screen.findByText(/test connection/i);
      fireEvent.click(testButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Connection Failed",
            variant: "destructive",
          })
        );
      });
    });
  });

  describe("Load Data", () => {
    it("should call runSync when Load Data is clicked", async () => {
      mockedApi.runSync.mockResolvedValue({
        detail: "Sync successful",
        dataset: {},
        stats: {},
      });

      await renderConnectionInfoPanel({ initialSettings: defaultInitialSettings });

      const loadButton = await screen.findByText(/sync data now/i);
      fireEvent.click(loadButton);

      await waitFor(() => {
        expect(mockedApi.runSync).toHaveBeenCalledWith(
          expect.objectContaining({
            hostname: "http://test.com",
            api_key: "test-key",
            mode: "incremental",
          })
        );
      });
    });
  });

  describe("Scheduler Drawer", () => {
    it("should open scheduler drawer when Scheduler button is clicked", async () => {
      await renderConnectionInfoPanel();

      const schedulerButton = await screen.findByText(/^scheduler$/i);
      fireEvent.click(schedulerButton);

      await waitFor(() => {
        expect(screen.getByText(/automatic scheduling/i)).toBeInTheDocument();
      });
    });

    it("should toggle scheduler when ON/OFF button is clicked", async () => {
      mockedApi.updateSyncScheduler.mockResolvedValue({
        enabled: true,
        interval_minutes: 60,
        last_run_at: null,
        next_run_at: null,
      });

      await renderConnectionInfoPanel();

      // Open drawer
      const schedulerButton = await screen.findByText(/^scheduler$/i);
      fireEvent.click(schedulerButton);

      await waitFor(() => screen.getByText(/automatic scheduling/i));

      // Click toggle
      const toggleButton = screen.getByText(/^off$/i);
      fireEvent.click(toggleButton);

      await waitFor(() => {
        expect(mockedApi.updateSyncScheduler).toHaveBeenCalledWith({
          enabled: true,
        });
      });
    });
  });
});
